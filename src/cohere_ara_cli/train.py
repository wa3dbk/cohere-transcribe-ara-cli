"""Fine-tuning Cohere Transcribe on TSV manifests (single GPU or multi-GPU DDP on one machine)."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import re
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    AudioDataset,
    DecodeCollator,
    DurationBatchSampler,
    Segment,
    TrainCollator,
    fill_durations,
    filter_segments,
    read_manifest,
    worker_init_fn,
    write_manifest,
)
from .decode import Transcriber
from .metrics import score
from .modeling import (
    LORA_TARGETS,
    count_parameters,
    eos_token_id,
    get_prompt_ids,
    hf_kwargs,
    inference_prefix,
    load_model,
    load_processor,
    pad_token_id,
    save_model_for_inference,
)
from .specaug import spec_augment
from .text import get_normalizer
from .utils import accelerate_kwargs, check_world

logger = logging.getLogger(__name__)

CKPT_RE = re.compile(r"checkpoint-(\d+)$")


# --------------------------------------------------------------------------------------
# Data preparation (cached, computed once on the main process)
# --------------------------------------------------------------------------------------
def prepare_split(accelerator, args, name: str, manifest: str) -> list[Segment]:
    cache = os.path.join(args.exp_dir, "data", f"{name}.tsv")
    if accelerator.is_main_process and (args.refresh_data_cache or not os.path.exists(cache)):
        segs = read_manifest(
            manifest, id_col=args.id_col, audio_col=args.audio_col, text_col=args.text_col,
            duration_col=args.duration_col, audio_root=args.audio_root, header=args.header, require_text=True,
        )
        bad = set(fill_durations(segs, num_threads=args.io_threads))
        if bad:
            logger.warning("%s: dropping %d unreadable audio files (e.g. %s)", name, len(bad), sorted(bad)[:3])
        write_manifest(cache, [s for s in segs if s.id not in bad])
    accelerator.wait_for_everyone()
    return read_manifest(cache, header="yes")


def tokenize_and_filter(segs: list[Segment], tokenizer, args, max_tokens: int, name: str, text_norm):
    texts = [text_norm(s.text or "") for s in segs]
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
    keep_segs, keep_ids = [], []
    stats = {"empty_text": 0, "too_many_tokens": 0, "too_dense": 0}
    for s, t in zip(segs, ids):
        if not t:
            stats["empty_text"] += 1
        elif len(t) > max_tokens:
            stats["too_many_tokens"] += 1
        elif s.duration and len(t) / s.duration > args.max_tokens_per_second:
            stats["too_dense"] += 1
        else:
            keep_segs.append(s)
            keep_ids.append(t)
    kept2, dstats = filter_segments(keep_segs, args.min_duration, args.max_duration)
    keep_set = {s.id for s in kept2}
    final_ids = [t for s, t in zip(keep_segs, keep_ids) if s.id in keep_set]
    stats.update(dstats)
    hours = sum(s.duration for s in kept2) / 3600
    ntok = sum(len(t) for t in final_ids)
    logger.info(
        "%s: kept %d/%d utterances (%.2f h, %d tokens, %.1f tok/s); filtered: %s",
        name, len(kept2), len(segs), hours, ntok, ntok / max(hours * 3600, 1e-6), stats,
    )
    return kept2, final_ids, {"kept": len(kept2), "total": len(segs), "hours": round(hours, 3), **stats}


# --------------------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------------------
def build_model(args, kw):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, **kw)
    enc = config.encoder_config
    if args.dropout is not None:
        enc.dropout = args.dropout
        enc.activation_dropout = args.dropout
    if args.attention_dropout is not None:
        enc.attention_dropout = args.attention_dropout
        config.attention_dropout = args.attention_dropout
    if args.layerdrop:
        enc.layerdrop = args.layerdrop
    config.use_cache = False
    # Trainable weights need fp32 masters; a frozen LoRA base does not (peft keeps adapters in fp32).
    load_dtype = torch.float32
    if args.lora and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    logger.info("Loading model weights in %s", load_dtype)
    model = load_model(args.model, dtype=load_dtype, attn_implementation=args.attn_implementation, config=config, **kw)

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad_(False)
    elif args.freeze_encoder_layers > 0:
        enc_mod = model.model.encoder
        for p in enc_mod.subsampling.parameters():
            p.requires_grad_(False)
        for layer in enc_mod.layers[: args.freeze_encoder_layers]:
            for p in layer.parameters():
                p.requires_grad_(False)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.lora:
        from peft import LoraConfig, get_peft_model

        parts = [p.strip() for p in args.lora_targets.split(",") if p.strip()]
        unknown = set(parts) - set(LORA_TARGETS)
        if unknown:
            raise ValueError(f"Unknown --lora-targets {unknown}; choose from {sorted(LORA_TARGETS)}")
        target = "|".join(f"(?:{LORA_TARGETS[p]})" for p in parts)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=target, bias="none")
        model = get_peft_model(model, lcfg)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
    return model


def build_optimizer(model, args):
    groups: dict[tuple[bool, bool], list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_enc = ".encoder." in n
        no_decay = p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") or "pos_emb" in n
        groups.setdefault((is_enc, no_decay), []).append(p)
    param_groups = []
    for (is_enc, no_decay), params in groups.items():
        param_groups.append({
            "params": params,
            "lr": args.lr * (args.encoder_lr_scale if is_enc else 1.0),
            "weight_decay": 0.0 if no_decay else args.weight_decay,
        })
    betas = (args.adam_beta1, args.adam_beta2)
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(param_groups, lr=args.lr, betas=betas, eps=args.adam_eps)
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(param_groups, lr=args.lr, betas=betas, eps=args.adam_eps, fused=fused)


# --------------------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------------------
def list_checkpoints(exp_dir: str) -> list[str]:
    d = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(d):
        return []
    found = []
    for name in os.listdir(d):
        m = CKPT_RE.search(name)
        if m and os.path.exists(os.path.join(d, name, "trainer_state.json")):
            found.append((int(m.group(1)), os.path.join(d, name)))
    return [p for _, p in sorted(found)]


def save_checkpoint(accelerator, args, tstate: dict) -> None:
    path = os.path.join(args.exp_dir, "checkpoints", f"checkpoint-{tstate['global_step']}")
    accelerator.save_state(path)
    if accelerator.is_main_process:
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump(tstate, f, indent=2)
        for old in list_checkpoints(args.exp_dir)[: -args.keep_checkpoints] if args.keep_checkpoints > 0 else []:
            shutil.rmtree(old, ignore_errors=True)
    accelerator.wait_for_everyone()


def save_inference_model(accelerator, model, processor, args, out_dir: str, merge_lora: bool = False) -> None:
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        if args.lora and not merge_lora:
            unwrapped.save_pretrained(out_dir)  # adapter only
            processor.save_pretrained(out_dir)
            with open(os.path.join(out_dir, "BASE_MODEL.txt"), "w") as f:
                f.write(args.model + "\n")
        else:
            m = unwrapped.merge_and_unload() if args.lora else unwrapped
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.save_dtype]
            m.config.use_cache = True
            save_model_for_inference(m, processor, out_dir, dtype)
            m.config.use_cache = False
    accelerator.wait_for_everyone()


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def validate(accelerator, model, loader, dev_decode, tr: Transcriber | None, args, normalizer) -> dict:
    net = accelerator.unwrap_model(model)  # avoid DDP collectives with uneven per-rank batch counts
    was_training = net.training
    net.eval()
    dev = accelerator.device
    loss_sum = torch.zeros((), device=dev, dtype=torch.float64)
    ntok = torch.zeros((), device=dev, dtype=torch.float64)
    for batch in loader:
        if not batch["indices"]:
            continue
        labels = batch["labels"].to(dev)
        out = net(
            input_features=batch["input_features"].to(dev, non_blocking=True),
            attention_mask=batch["attention_mask"].to(dev, non_blocking=True),
            decoder_input_ids=batch["decoder_input_ids"].to(dev, non_blocking=True),
            use_cache=False,
        )
        logits = out.logits.float()
        loss_sum += F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum").double()
        ntok += (labels != -100).sum().double()
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    ntok = accelerator.reduce(ntok, reduction="sum")
    result = {"valid_loss": (loss_sum / ntok.clamp(min=1)).item()}

    if tr is not None and dev_decode is not None:
        from accelerate.utils import gather_object

        segs, dec_loader = dev_decode
        tr.model = net
        net.config.use_cache = True
        local = []
        for batch in dec_loader:
            if batch["indices"]:
                hyps = tr.transcribe_batch(batch)
                local.extend((segs[i].id, segs[i].text or "", h) for i, h in zip(batch["indices"], hyps))
        net.config.use_cache = False
        all_items = gather_object(local)
        if accelerator.is_main_process:
            rep, _ = score(all_items, normalizer, args.valid_normalizer, keep_utts=False)
            result["valid_wer"] = 100 * rep.wer
            result["valid_cer"] = 100 * rep.cer
            examples = all_items[: args.valid_log_examples]
            for uid, ref, hyp in examples:
                logger.info("  [%s]\n    REF: %s\n    HYP: %s", uid, ref, hyp)
        obj = [result.get("valid_wer"), result.get("valid_cer")]
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(obj, from_process=0)
        result["valid_wer"], result["valid_cer"] = obj
    if was_training:
        net.train()
    return result


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------
def resolve_mixed_precision(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "no"


def run_train(args) -> dict | None:
    from accelerate import Accelerator, DistributedType
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from transformers import get_scheduler

    os.makedirs(args.exp_dir, exist_ok=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(args.find_unused_parameters or args.layerdrop > 0))
    report_to = None if args.report_to == "none" else args.report_to
    accelerator = Accelerator(
        mixed_precision=resolve_mixed_precision(args.mixed_precision),
        gradient_accumulation_steps=1,  # accumulation handled manually (exact control with dynamic batches)
        log_with=report_to,
        project_dir=args.exp_dir,
        kwargs_handlers=[ddp_kwargs],
        **accelerate_kwargs(),
    )
    check_world(accelerator.state)
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.MULTI_CPU):
        raise SystemExit(f"Distributed type {accelerator.distributed_type} is not supported; use DDP (torchrun / --gpus).")
    if accelerator.is_main_process:
        fh = logging.FileHandler(os.path.join(args.exp_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
        with open(os.path.join(args.exp_dir, "train_args.json"), "w") as f:
            json.dump({k: v for k, v in vars(args).items() if k != "func"}, f, indent=2, default=str)
    set_seed(args.seed)
    logger.info("Accelerator: %s, processes=%d, mixed_precision=%s", accelerator.distributed_type,
                accelerator.num_processes, accelerator.mixed_precision)

    kw = hf_kwargs(args.revision, args.hf_token, args.cache_dir)
    processor = load_processor(args.processor or args.model, **kw)
    model = build_model(args, kw)
    base = model.get_base_model() if args.lora else model
    prompt_ids = get_prompt_ids(processor, args.language, not args.no_punctuation)
    prefix = inference_prefix(base, prompt_ids)
    eos_id = eos_token_id(base)
    pad_id = pad_token_id(base, processor)
    max_tokens = base.config.max_position_embeddings - len(prefix) - 1
    if args.max_tokens:
        max_tokens = min(max_tokens, args.max_tokens)
    total, trainable = count_parameters(model)
    logger.info("Parameters: %.1fM total, %.1fM trainable (%.2f%%)", total / 1e6, trainable / 1e6, 100 * trainable / total)
    logger.info("Teacher-forcing prefix %s, eos=%d, pad=%d, max label tokens=%d", prefix, eos_id, pad_id, max_tokens)

    # ---------------- data
    text_norm = get_normalizer(args.train_text_normalizer)
    train_segs = prepare_split(accelerator, args, "train", args.train_manifest)
    train_segs, train_ids, train_stats = tokenize_and_filter(train_segs, processor.tokenizer, args, max_tokens, "train", text_norm)
    if not train_segs:
        raise SystemExit("No training data left after filtering")
    data_stats = {"train": train_stats}

    sampler = DurationBatchSampler(
        [s.duration for s in train_segs], args.max_batch_duration, args.batch_size, shuffle=True, seed=args.seed,
        num_replicas=accelerator.num_processes, rank=accelerator.process_index,
    )
    collator = TrainCollator(processor.feature_extractor, prefix, eos_id, pad_id)
    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(AudioDataset(train_segs, train_ids), batch_sampler=sampler, collate_fn=collator, **loader_kw)

    valid_loader = dev_decode = None
    if args.valid_manifest:
        dev_segs = prepare_split(accelerator, args, "valid", args.valid_manifest)
        dev_segs, dev_ids, dev_stats = tokenize_and_filter(dev_segs, processor.tokenizer, args, max_tokens, "valid", text_norm)
        data_stats["valid"] = dev_stats
        vsampler = DurationBatchSampler([s.duration for s in dev_segs], args.valid_max_batch_duration or args.max_batch_duration,
                                        args.batch_size, shuffle=False, num_replicas=accelerator.num_processes,
                                        rank=accelerator.process_index)
        vkw = dict(loader_kw, persistent_workers=False)
        valid_loader = DataLoader(AudioDataset(dev_segs, dev_ids), batch_sampler=vsampler, collate_fn=collator, **vkw)
        if args.valid_decode_utts != 0:
            rng = random.Random(args.seed)
            pick = list(range(len(dev_segs)))
            if 0 < args.valid_decode_utts < len(dev_segs):
                pick = sorted(rng.sample(pick, args.valid_decode_utts))
            dsegs = [dev_segs[i] for i in pick]
            dsampler = DurationBatchSampler([s.duration for s in dsegs], args.valid_max_batch_duration or args.max_batch_duration,
                                            args.batch_size, shuffle=False, num_replicas=accelerator.num_processes,
                                            rank=accelerator.process_index)
            dloader = DataLoader(AudioDataset(dsegs), batch_sampler=dsampler,
                                 collate_fn=DecodeCollator(processor.feature_extractor), **vkw)
            dev_decode = (dsegs, dloader)
    if accelerator.is_main_process:
        with open(os.path.join(args.exp_dir, "data", "stats.json"), "w") as f:
            json.dump(data_stats, f, indent=2)

    # ---------------- optimization
    optimizer = build_optimizer(model, args)
    batches_per_epoch = len(sampler)
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = args.max_steps or args.num_epochs * updates_per_epoch
    warmup = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_ratio * total_steps)
    scheduler = get_scheduler(args.lr_scheduler, optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    logger.info("Batches/epoch/rank=%d, updates/epoch=%d, total updates=%d, warmup=%d",
                batches_per_epoch, updates_per_epoch, total_steps, warmup)

    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    if report_to:
        accelerator.init_trackers("cohere-ara", config={k: str(v) for k, v in vars(args).items() if k != "func"})

    dmodel = accelerator.unwrap_model(model)
    tr = None
    if dev_decode is not None:
        gen = {"do_sample": False, "num_beams": 1}
        tr = Transcriber(dmodel, processor, args.language, not args.no_punctuation, gen, accelerator.device,
                         torch.float32, args.valid_max_new_tokens, 20.0)
    normalizer = get_normalizer(args.valid_normalizer)
    metric_name = "valid_wer" if dev_decode is not None else "valid_loss"

    # ---------------- resume
    tstate = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0, "best_metric": None, "best_step": None,
              "evals_without_improvement": 0, "history": []}
    ckpt = args.resume_from
    if ckpt is None and args.resume:
        cks = list_checkpoints(args.exp_dir)
        ckpt = cks[-1] if cks else None
    if ckpt:
        logger.info("Resuming from %s", ckpt)
        accelerator.load_state(ckpt)
        with open(os.path.join(ckpt, "trainer_state.json")) as f:
            tstate.update(json.load(f))
        if tstate.get("metric") not in (None, metric_name):
            logger.warning("Best-model metric changed (%s -> %s): resetting best", tstate["metric"], metric_name)
            tstate.update(best_metric=None, best_step=None, evals_without_improvement=0)
    tstate["metric"] = metric_name

    def do_eval(tag: str) -> bool:
        """Run validation, maybe save best. Returns True if early stopping triggers."""
        if valid_loader is None:
            return False
        t = time.time()
        res = validate(accelerator, model, valid_loader, dev_decode, tr, args, normalizer)
        res.update(step=tstate["global_step"], epoch=tstate["epoch"], tag=tag)
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items())
        logger.info("Validation (%s) %s [%.0fs]", tag, msg, time.time() - t)
        if report_to:
            accelerator.log({k: v for k, v in res.items() if isinstance(v, (int, float))}, step=tstate["global_step"])
        tstate["history"].append(res)
        cur = res.get(metric_name)
        if cur is None or tag == "start":
            return False
        if tstate["best_metric"] is None or cur < tstate["best_metric"] - 1e-9:
            tstate.update(best_metric=cur, best_step=tstate["global_step"], evals_without_improvement=0)
            logger.info("New best %s=%.4f at step %d -> saving %s", metric_name, cur, tstate["global_step"],
                        os.path.join(args.exp_dir, "best"))
            save_inference_model(accelerator, model, processor, args, os.path.join(args.exp_dir, "best"))
        else:
            tstate["evals_without_improvement"] += 1
        return bool(args.early_stopping_patience and tstate["evals_without_improvement"] >= args.early_stopping_patience)

    if args.eval_at_start and tstate["global_step"] == 0:
        do_eval("start")

    # ---------------- training loop
    model.train()
    stop = tstate["global_step"] >= total_steps
    early_stopped = False
    last_good = None
    log_loss = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    log_tok = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    log_audio = 0.0
    t_log = time.time()
    start_epoch = tstate["epoch"]
    while not stop and tstate["epoch"] < args.num_epochs:
        epoch = tstate["epoch"]
        sampler.set_epoch(epoch)
        skip = tstate["batch_in_epoch"] if epoch == start_epoch else 0
        if skip:
            sampler.set_start(skip)
            logger.info("Epoch %d: skipping %d already-seen batches", epoch, skip)
        n_batches = len(sampler)
        batch_i = skip
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch_i += 1
            micro += 1
            dummy = False
            if not batch["indices"]:
                if last_good is None:
                    raise RuntimeError("First training batch has no readable audio")
                batch, dummy = last_good, True  # keep DDP ranks in lock-step; contributes zero gradient
            else:
                last_good = batch
            sync = micro == args.grad_accum or batch_i == n_batches
            ctx = contextlib.nullcontext() if sync else accelerator.no_sync(model)
            with ctx:
                dev = accelerator.device
                feats = batch["input_features"].to(dev, non_blocking=True)
                amask = batch["attention_mask"].to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True)
                if args.spec_augment:
                    feats = spec_augment(feats, amask, args.freq_masks, args.freq_mask_width, args.time_masks,
                                         args.time_mask_ratio, args.max_time_mask_width)
                out = model(input_features=feats, attention_mask=amask,
                            decoder_input_ids=batch["decoder_input_ids"].to(dev, non_blocking=True), use_cache=False)
                logits = out.logits.float()
                loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
                                           reduction="sum", label_smoothing=args.label_smoothing)
                ntok = (labels != -100).sum()
                loss = loss_sum / ntok.clamp(min=1) / args.grad_accum
                if dummy:
                    loss = loss * 0.0
                accelerator.backward(loss)
            if not dummy:
                log_loss += loss_sum.detach().double()
                log_tok += ntok.double()
                log_audio += batch["audio_seconds"]
            if not sync:
                continue
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm) if args.max_grad_norm > 0 else None
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            tstate["global_step"] += 1
            tstate["batch_in_epoch"] = batch_i
            step = tstate["global_step"]

            if step % args.log_interval == 0:
                tl = accelerator.reduce(log_loss.clone(), reduction="sum")
                tt = accelerator.reduce(log_tok.clone(), reduction="sum")
                ta = accelerator.reduce(torch.tensor(log_audio, device=accelerator.device, dtype=torch.float64), reduction="sum")
                el = time.time() - t_log
                rec = {
                    "train_loss": (tl / tt.clamp(min=1)).item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
                    "audio_h_per_h": ta.item() / max(el, 1e-6),
                    "epoch": epoch + batch_i / max(n_batches, 1),
                }
                mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if torch.cuda.is_available() else ""
                logger.info("step %d/%d ep %.2f loss %.4f lr %.2e gnorm %.2f speed %.0fx realtime%s", step, total_steps,
                            rec["epoch"], rec["train_loss"], rec["lr"], rec["grad_norm"], rec["audio_h_per_h"], mem)
                if report_to:
                    accelerator.log(rec, step=step)
                log_loss.zero_()
                log_tok.zero_()
                log_audio = 0.0
                t_log = time.time()

            if args.valid_interval and step % args.valid_interval == 0:
                if do_eval(f"step{step}"):
                    logger.info("Early stopping (no improvement for %d evaluations)", args.early_stopping_patience)
                    early_stopped = stop = True
            if args.save_interval and step % args.save_interval == 0:
                save_checkpoint(accelerator, args, tstate)
            if step >= total_steps:
                stop = True
            if stop:
                break
        if batch_i >= n_batches:  # epoch fully consumed (also when the step budget ended exactly here)
            tstate["epoch"] = epoch + 1
            tstate["batch_in_epoch"] = 0
            logger.info("Finished epoch %d", epoch + 1)
            if not early_stopped and args.valid_every_epoch and do_eval(f"epoch{epoch + 1}"):
                logger.info("Early stopping (no improvement for %d evaluations)", args.early_stopping_patience)
                early_stopped = stop = True
            if args.save_every_epoch and not stop:
                save_checkpoint(accelerator, args, tstate)

    # ---------------- final
    if valid_loader is not None and (not tstate["history"] or tstate["history"][-1].get("step") != tstate["global_step"]):
        do_eval("final")
    save_checkpoint(accelerator, args, tstate)
    final_dir = os.path.join(args.exp_dir, "final")
    save_inference_model(accelerator, model, processor, args, final_dir, merge_lora=args.lora and args.merge_lora_at_end)
    if valid_loader is None and accelerator.is_main_process:
        logger.info("No validation set: 'final' is the model to use.")
    if report_to:
        accelerator.end_training()
    summary = {k: tstate[k] for k in ("global_step", "epoch", "best_metric", "best_step")}
    summary["metric"] = metric_name
    summary["final_dir"] = final_dir
    summary["best_dir"] = os.path.join(args.exp_dir, "best") if tstate["best_step"] is not None else None
    if accelerator.is_main_process:
        with open(os.path.join(args.exp_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Done: %s", summary)
    return summary
