import os

import numpy as np
import pytest

from cohere_transcribe_ara_cli.data import (
    DurationBatchSampler,
    Segment,
    TrainCollator,
    build_decoder_sequences,
    fill_durations,
    load_audio,
    pack_batches,
    read_manifest,
    write_manifest,
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_manifest_no_header(tmp_path, wav_factory):
    wav_factory("a.wav", 1.0)
    p = tmp_path / "m.tsv"
    _write(p, 'seg1\ta.wav\tقال "مرحبا" لي\n')  # quotes must survive (QUOTE_NONE)
    segs = read_manifest(str(p))
    assert segs[0].id == "seg1" and segs[0].text == 'قال "مرحبا" لي'
    assert segs[0].audio == os.path.join(str(tmp_path), "a.wav")


def test_manifest_common_voice_header(tmp_path):
    p = tmp_path / "cv.tsv"
    _write(p, "client_id\tpath\tsentence\tup_votes\nabc\tclip1.wav\tنص\t2\n")
    segs = read_manifest(str(p), audio_root="/data/clips")
    assert segs[0].id == "clip1" and segs[0].audio == "/data/clips/clip1.wav" and segs[0].text == "نص"


def test_manifest_explicit_columns(tmp_path):
    p = tmp_path / "m.tsv"
    _write(p, "x\t/abs/a.wav\t3.5\tمرحبا\n")
    segs = read_manifest(str(p), id_col="0", audio_col="1", text_col="3", duration_col="2")
    assert segs[0].duration == 3.5 and segs[0].text == "مرحبا"


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "m.tsv"
    _write(p, "a\t/x.wav\tt\na\t/y.wav\tt\n")
    with pytest.raises(ValueError):
        read_manifest(str(p))


def test_write_read_roundtrip(tmp_path):
    segs = [Segment("a", "/x.wav", "نص\tمع", 1.234)]
    p = str(tmp_path / "o.tsv")
    write_manifest(p, segs)
    back = read_manifest(p, header="yes")
    assert back[0].duration == pytest.approx(1.234) and back[0].text == "نص مع"


def test_audio_resample_and_downmix(wav_factory):
    path = wav_factory("s.wav", 0.5, sr=8000, channels=2)
    y = load_audio(path)
    assert y.ndim == 1 and abs(len(y) - 8000) <= 2 and y.dtype == np.float32
    segs = [Segment("s", path), Segment("bad", "/does/not/exist.wav")]
    assert fill_durations(segs, show_progress=False) == ["bad"]
    assert segs[0].duration == pytest.approx(0.5)


def test_pack_batches_padded_cost():
    durs = [10.0, 9.0, 1.0, 1.0, 1.0]
    batches = pack_batches([0, 1, 2, 3, 4], durs, max_batch_duration=20.0, max_batch_size=None)
    assert batches == [[0, 1], [2, 3, 4]]
    for b in batches:
        assert len(b) * max(durs[i] for i in b) <= 20.0


def test_sampler_ddp_properties():
    rng = np.random.default_rng(0)
    durs = list(rng.uniform(1, 20, size=500))
    samplers = [DurationBatchSampler(durs, 60, 16, shuffle=True, seed=3, num_replicas=3, rank=r) for r in range(3)]
    for s in samplers:
        s.set_epoch(2)
    per_rank = [list(s) for s in samplers]
    assert len({len(b) for b in per_rank}) == 1                       # equal #batches (DDP)
    flat = [i for bs in per_rank for b in bs for i in b]
    assert len(flat) == len(set(flat))                                  # disjoint
    again = DurationBatchSampler(durs, 60, 16, shuffle=True, seed=3, num_replicas=3, rank=0)
    again.set_epoch(2)
    assert list(again) == per_rank[0]                                   # deterministic
    again.set_epoch(3)
    assert list(again) != per_rank[0]                                   # reshuffled per epoch
    again.set_epoch(2)
    again.set_start(4)
    assert list(again) == per_rank[0][4:]                               # resume skipping
    assert list(again) == per_rank[0]                                   # skipping is one-shot


def test_sampler_decode_covers_everything():
    durs = [float(i % 7 + 1) for i in range(101)]
    s0 = DurationBatchSampler(durs, 30, 8, num_replicas=2, rank=0)
    s1 = DurationBatchSampler(durs, 30, 8, num_replicas=2, rank=1)
    got = sorted(i for s in (s0, s1) for b in s for i in b)
    assert got == list(range(101))


def test_decoder_sequences_alignment():
    prefix, text, eos = [4, 10, 11], [50, 51], 3
    dec_in, labels = build_decoder_sequences(prefix, text, eos)
    assert dec_in == [4, 10, 11, 50, 51]
    assert labels == [-100, -100, 50, 51, 3]
    # position i of the decoder input predicts labels[i]
    assert dec_in[2] == prefix[-1] and labels[2] == text[0]


def test_train_collator_padding(tiny):
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(tiny["model"])
    col = TrainCollator(proc.feature_extractor, prefix=[4, 5], eos_id=3, pad_id=2)
    items = [
        {"idx": 0, "audio": np.zeros(16000, np.float32) + 0.01, "labels": [7, 8, 9]},
        {"idx": 1, "audio": np.zeros(8000, np.float32) + 0.01, "labels": [7]},
        {"idx": 2, "error": "boom", "labels": [1]},
    ]
    b = col(items)
    assert b["indices"] == [0, 1] and b["failed"] == [(2, "boom")]
    assert b["decoder_input_ids"].tolist() == [[4, 5, 7, 8, 9], [4, 5, 7, 2, 2]]
    assert b["labels"].tolist() == [[-100, 7, 8, 9, 3], [-100, 7, 3, -100, -100]]
    assert b["input_features"].shape[0] == 2 and b["attention_mask"].dtype.is_floating_point is False
