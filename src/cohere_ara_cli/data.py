"""Data handling: Common-Voice-like TSV manifests, audio I/O, duration-based batching, collators."""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import sys
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from . import SAMPLE_RATE

logger = logging.getLogger(__name__)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ID_COLUMNS = ("id", "segment_id", "seg_id", "utt_id", "uttid", "utterance_id", "key", "segment")
AUDIO_COLUMNS = ("path", "audio", "audio_path", "audio_filepath", "wav", "wav_path", "file", "filename", "audio_file")
TEXT_COLUMNS = ("sentence", "text", "transcript", "transcription", "normalized_text", "ref", "reference")
DURATION_COLUMNS = ("duration", "dur", "duration_s", "duration_sec", "seconds")


@dataclass
class Segment:
    id: str
    audio: str
    text: str | None = None
    duration: float | None = None


# --------------------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------------------
def _resolve_col(spec: str | None, header: list[str] | None, candidates: Sequence[str], default_idx: int | None):
    """Resolve a column spec (name or 0-based index) to an index; None if absent."""
    if spec is not None:
        if spec.isdigit():
            return int(spec)
        if header is None:
            raise ValueError(f"Column name {spec!r} given but the manifest has no header row")
        lowered = [h.lower() for h in header]
        if spec.lower() not in lowered:
            raise ValueError(f"Column {spec!r} not found in header {header}")
        return lowered.index(spec.lower())
    if header is not None:
        lowered = [h.strip().lower() for h in header]
        for c in candidates:
            if c in lowered:
                return lowered.index(c)
        return None
    return default_idx


def _looks_like_header(fields: list[str]) -> bool:
    known = set(ID_COLUMNS + AUDIO_COLUMNS + TEXT_COLUMNS + DURATION_COLUMNS + ("client_id",))
    return any(f.strip().lower() in known for f in fields)


def read_manifest(
    path: str,
    id_col: str | None = None,
    audio_col: str | None = None,
    text_col: str | None = None,
    duration_col: str | None = None,
    audio_root: str | None = None,
    header: str = "auto",
    require_text: bool = False,
) -> list[Segment]:
    """Read a TSV manifest.

    Without a header, columns default to: 0=id, 1=audio path, 2=transcript (optional), 3=duration (only
    used when --duration-col is given). With a header, common Common Voice / NeMo style names are detected.
    Relative audio paths are resolved against ``audio_root`` (or the manifest's directory).
    """
    root = audio_root if audio_root is not None else os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = [r for r in rows if r and any(x.strip() for x in r)]
    if not rows:
        raise ValueError(f"Manifest {path} is empty")

    if header == "auto":
        has_header = _looks_like_header(rows[0])
    else:
        has_header = header == "yes"
    hdr = [h.strip() for h in rows[0]] if has_header else None
    body = rows[1:] if has_header else rows

    i_audio = _resolve_col(audio_col, hdr, AUDIO_COLUMNS, 1)
    if i_audio is None:
        raise ValueError(f"Could not find an audio-path column in {hdr}; use --audio-col")
    i_id = _resolve_col(id_col, hdr, ID_COLUMNS, 0)
    i_text = _resolve_col(text_col, hdr, TEXT_COLUMNS, 2)
    i_dur = _resolve_col(duration_col, hdr, DURATION_COLUMNS, None)

    segments: list[Segment] = []
    seen: set[str] = set()
    skipped = 0
    for lineno, r in enumerate(body, start=2 if has_header else 1):
        if i_audio >= len(r):
            skipped += 1
            continue
        audio = r[i_audio].strip()
        if not os.path.isabs(audio):
            audio = os.path.abspath(os.path.join(root, audio))
        uid = r[i_id].strip() if (i_id is not None and i_id < len(r)) else os.path.splitext(os.path.basename(audio))[0]
        text = r[i_text].strip() if (i_text is not None and i_text < len(r)) else None
        if require_text and text is None:
            skipped += 1
            continue
        dur = None
        if i_dur is not None and i_dur < len(r) and r[i_dur].strip():
            try:
                dur = float(r[i_dur])
            except ValueError:
                dur = None
        if uid in seen:
            raise ValueError(f"Duplicate segment id {uid!r} at line {lineno} of {path}")
        seen.add(uid)
        segments.append(Segment(uid, audio, text, dur))
    if skipped:
        logger.warning("Skipped %d malformed/incomplete rows in %s", skipped, path)
    logger.info(
        "Read %d segments from %s (header=%s, id=%s, audio=%s, text=%s, duration=%s)",
        len(segments), path, has_header, i_id, i_audio, i_text, i_dur,
    )
    return segments


def write_manifest(path: str, segments: Sequence[Segment]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("id\tpath\tsentence\tduration\n")
        for s in segments:
            text = (s.text or "").replace("\t", " ").replace("\n", " ")
            dur = f"{s.duration:.3f}" if s.duration is not None else ""
            f.write(f"{s.id}\t{s.audio}\t{text}\t{dur}\n")


# --------------------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------------------
def audio_info(path: str) -> tuple[float, int, int]:
    """(duration_s, sample_rate, channels) read from the file header only."""
    info = sf.info(path)
    return info.frames / float(info.samplerate), int(info.samplerate), int(info.channels)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load mono float32 audio at ``target_sr`` (downmix + resample if needed)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(data, dtype=np.float32)


def fill_durations(segments: list[Segment], num_threads: int = 16, show_progress: bool = True) -> list[str]:
    """Fill missing durations in-place by reading file headers in parallel. Returns ids of unreadable files."""
    todo = [s for s in segments if s.duration is None]
    if not todo:
        return []
    bad: list[str] = []

    def _one(seg: Segment):
        try:
            return audio_info(seg.audio)[0]
        except Exception:  # noqa: BLE001
            return None

    it = None
    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as ex:
        it = ex.map(_one, todo, chunksize=64)
        if show_progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(todo), desc="Reading audio headers", unit="file", mininterval=2.0)
        for seg, dur in zip(todo, it):
            if dur is None:
                bad.append(seg.id)
            seg.duration = dur
    return bad


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class AudioDataset(Dataset):
    """Loads raw audio for a list of segments; errors are returned, not raised (so a bad file doesn't kill a run)."""

    def __init__(self, segments: Sequence[Segment], labels: Sequence[list[int]] | None = None):
        self.segments = segments
        self.labels = labels

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> dict:
        seg = self.segments[idx]
        item: dict = {"idx": idx}
        try:
            item["audio"] = load_audio(seg.audio)
        except Exception as e:  # noqa: BLE001
            item["error"] = f"{type(e).__name__}: {e}"
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def worker_init_fn(_worker_id: int) -> None:
    # Feature extraction runs inside workers: keep each worker single-threaded to avoid oversubscription.
    torch.set_num_threads(1)


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------
def pack_batches(
    order: Sequence[int],
    durations: Sequence[float],
    max_batch_duration: float,
    max_batch_size: int | None,
) -> list[list[int]]:
    """Greedy packing of an ordering by *padded* cost = n_items * max_duration_in_batch."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0.0
    for i in order:
        d = durations[i]
        new_max = max(cur_max, d)
        too_big = cur and (new_max * (len(cur) + 1) > max_batch_duration)
        too_many = max_batch_size is not None and len(cur) >= max_batch_size
        if too_big or too_many:
            batches.append(cur)
            cur, new_max = [], d
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


class DurationBatchSampler(Sampler[list[int]]):
    """Dynamic batch sampler: batches hold up to ``max_batch_duration`` seconds of *padded* audio.

    * ``shuffle=False`` (decoding): sort by decreasing duration (largest batch first -> OOM surfaces early),
      batches are dealt round-robin to ranks.
    * ``shuffle=True`` (training): sort by duration perturbed with multiplicative noise (bucketing with
      randomness), pack, shuffle batch order with an epoch seed, and give every rank the same number of
      batches (required by DDP).  ``set_epoch`` must be called each epoch; ``set_start`` skips batches
      when resuming mid-epoch.
    """

    def __init__(
        self,
        durations: Sequence[float],
        max_batch_duration: float,
        max_batch_size: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        bucket_noise: float = 0.1,
    ):
        if max_batch_duration <= 0:
            raise ValueError("max_batch_duration must be > 0")
        self.durations = list(durations)
        self.max_batch_duration = max_batch_duration
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_noise = bucket_noise
        self.epoch = 0
        self.start = 0
        self._cache: tuple[int, list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start(self, start_batch: int) -> None:
        self.start = start_batch

    def _all_batches(self) -> list[list[int]]:
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]
        n = len(self.durations)
        if self.shuffle:
            rng = random.Random(self.seed + 1000003 * self.epoch)
            keys = [self.durations[i] * (1.0 + rng.uniform(-self.bucket_noise, self.bucket_noise)) for i in range(n)]
            order = sorted(range(n), key=lambda i: keys[i])
            batches = pack_batches(order, self.durations, self.max_batch_duration, self.max_batch_size)
            rng.shuffle(batches)
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            if usable == 0 and batches:
                raise ValueError(
                    f"Only {len(batches)} batches for {self.num_replicas} processes: dataset too small "
                    "or --max-batch-duration too large"
                )
            batches = batches[:usable]
        else:
            order = sorted(range(n), key=lambda i: -self.durations[i])
            batches = pack_batches(order, self.durations, self.max_batch_duration, self.max_batch_size)
        self._cache = (self.epoch, batches)
        return batches

    def rank_batches(self) -> list[list[int]]:
        return self._all_batches()[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.rank_batches()
        start, self.start = self.start, 0  # skipping applies to one iteration only
        yield from batches[start:]

    def __len__(self) -> int:
        return len(self.rank_batches())


# --------------------------------------------------------------------------------------
# Collators
# --------------------------------------------------------------------------------------
def _split_ok_failed(items: list[dict]) -> tuple[list[dict], list[tuple[int, str]]]:
    ok = [it for it in items if "audio" in it and it["audio"].size > 0]
    failed = [(it["idx"], it.get("error", "empty audio")) for it in items if not ("audio" in it and it["audio"].size > 0)]
    return ok, failed


class DecodeCollator:
    """Builds model inputs for generation. Long audio (> ~30 s) is chunked by the feature extractor."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, items: list[dict]) -> dict:
        ok, failed = _split_ok_failed(items)
        batch: dict = {"indices": [it["idx"] for it in ok], "failed": failed}
        if not ok:
            return batch
        feats = self.fe(
            [it["audio"] for it in ok],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        batch["input_features"] = feats["input_features"]
        batch["attention_mask"] = feats["attention_mask"]
        batch["audio_chunk_index"] = list(feats.get("audio_chunk_index") or [(i, None) for i in range(len(ok))])
        batch["audio_seconds"] = float(sum(it["audio"].shape[0] for it in ok)) / SAMPLE_RATE
        return batch


def build_decoder_sequences(
    prefix: Sequence[int], text_ids: Sequence[int], eos_id: int
) -> tuple[list[int], list[int]]:
    """Teacher-forcing pair for one utterance.

    full   = prefix + text + [eos]
    input  = full[:-1]
    labels = full[1:], with the positions that predict prompt tokens masked (-100)
    """
    full = list(prefix) + list(text_ids) + [eos_id]
    dec_in = full[:-1]
    labels = full[1:]
    for i in range(len(prefix) - 1):
        labels[i] = -100
    return dec_in, labels


class TrainCollator:
    """Features + teacher-forcing decoder inputs/labels (right padded)."""

    def __init__(self, feature_extractor, prefix: Sequence[int], eos_id: int, pad_id: int):
        self.fe = feature_extractor
        self.prefix = list(prefix)
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __call__(self, items: list[dict]) -> dict:
        ok, failed = _split_ok_failed(items)
        batch: dict = {"indices": [it["idx"] for it in ok], "failed": failed}
        if not ok:
            return batch
        feats = self.fe(
            [it["audio"] for it in ok],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        chunk_index = feats.get("audio_chunk_index")
        if chunk_index is not None and any(c is not None for _, c in chunk_index):
            raise RuntimeError(
                "An utterance longer than the model's single-chunk limit reached the training collator; "
                "lower --max-duration (<= 30 s)."
            )
        pairs = [build_decoder_sequences(self.prefix, it["labels"], self.eos_id) for it in ok]
        max_len = max(len(p[0]) for p in pairs)
        dec_in = torch.full((len(pairs), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(pairs), max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(pairs):
            dec_in[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            labels[i, : len(y)] = torch.tensor(y, dtype=torch.long)
        batch.update(
            input_features=feats["input_features"],
            attention_mask=feats["attention_mask"],
            decoder_input_ids=dec_in,
            labels=labels,
            audio_seconds=float(sum(it["audio"].shape[0] for it in ok)) / SAMPLE_RATE,
        )
        return batch


def estimate_num_batches(durations: Sequence[float], max_batch_duration: float, max_batch_size: int | None) -> int:
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    return len(pack_batches(order, durations, max_batch_duration, max_batch_size))


def filter_segments(
    segments: list[Segment],
    min_duration: float,
    max_duration: float,
    predicate: Callable[[Segment], bool] | None = None,
) -> tuple[list[Segment], dict[str, int]]:
    kept, stats = [], {"missing_duration": 0, "too_short": 0, "too_long": 0, "rejected": 0}
    for s in segments:
        if s.duration is None or math.isnan(s.duration):
            stats["missing_duration"] += 1
        elif s.duration < min_duration:
            stats["too_short"] += 1
        elif s.duration > max_duration:
            stats["too_long"] += 1
        elif predicate is not None and not predicate(s):
            stats["rejected"] += 1
        else:
            kept.append(s)
    return kept, stats
