"""End-to-end CLI tests on a tiny random model (CPU, no downloads)."""

import json
import os

import numpy as np
import pytest
import soundfile as sf

from cohere_transcribe_ara_cli.cli import main

pytestmark = pytest.mark.slow

PHRASE = "مرحبا بكم في تونس"


def _single_phrase_dataset(root: str, n: int = 12) -> str:
    rng = np.random.default_rng(1)
    os.makedirs(os.path.join(root, "w"), exist_ok=True)
    path = os.path.join(root, "one.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            wav = os.path.join(root, "w", f"u{i}.wav")
            sf.write(wav, (0.1 * rng.standard_normal(int(16000 * rng.uniform(0.5, 1.5)))).astype(np.float32), 16000)
            f.write(f"u{i}\t{wav}\t{PHRASE}\n")  # headerless: id, path, text
    return path


def test_decode_and_score(tiny, tmp_path):
    out = str(tmp_path / "dec")
    assert main(["decode", tiny["valid"], "-o", out, "--model", tiny["model"], "--num-workers", "0",
                 "--max-new-tokens", "8", "--details"]) == 0
    for f in ("hyp.tsv", "hyp.txt", "wer.json", "wer.txt", "summary.json", "details.arabic.txt"):
        assert os.path.exists(os.path.join(out, f)), f
    with open(os.path.join(out, "hyp.tsv"), encoding="utf-8") as f:
        assert len(f.readlines()) == 1 + 6
    wer = json.load(open(os.path.join(out, "wer.json")))
    assert set(wer) == {"arabic", "none"}

    # `score` on the produced hypotheses reproduces the same numbers
    out2 = str(tmp_path / "score")
    assert main(["score", "--ref", tiny["valid"], "--hyp", os.path.join(out, "hyp.tsv"), "-o", out2]) == 0
    assert json.load(open(os.path.join(out2, "wer.json")))["arabic"]["wer"] == wer["arabic"]["wer"]

    # resume: nothing left to do, identical output
    before = open(os.path.join(out, "hyp.tsv"), encoding="utf-8").read()
    assert main(["decode", tiny["valid"], "-o", out, "--model", tiny["model"], "--num-workers", "0",
                 "--max-new-tokens", "8", "--resume"]) == 0
    assert open(os.path.join(out, "hyp.tsv"), encoding="utf-8").read() == before


def test_overfit_then_exact_decode(tiny, tmp_path):
    """If prefix / label shift / EOS were wrong, greedy decoding could not reproduce the target exactly."""
    data = _single_phrase_dataset(str(tmp_path))
    exp = str(tmp_path / "exp")
    assert main(["train", "--train-manifest", data, "--valid-manifest", data, "--exp-dir", exp,
                 "--model", tiny["model"], "--max-steps", "60", "--num-epochs", "100", "--lr", "3e-3",
                 "--warmup-steps", "5", "--max-batch-duration", "10", "--num-workers", "0",
                 "--valid-interval", "0", "--no-valid-every-epoch", "--no-eval-at-start", "--save-interval", "0",
                 "--no-save-every-epoch", "--log-interval", "20", "--save-dtype", "fp32"]) == 0
    summary = json.load(open(os.path.join(exp, "train_summary.json")))
    assert summary["global_step"] == 60
    out = str(tmp_path / "dec")
    assert main(["decode", data, "-o", out, "--model", os.path.join(exp, "final"), "--num-workers", "0",
                 "--dtype", "fp32"]) == 0
    res = json.load(open(os.path.join(out, "summary.json")))
    assert res["wer_none"] == 0.0, open(os.path.join(out, "hyp.tsv"), encoding="utf-8").read()


def test_train_resume_and_lora(tiny, tmp_path):
    exp = str(tmp_path / "exp")
    common = ["train", "--train-manifest", tiny["train"], "--valid-manifest", tiny["valid"], "--exp-dir", exp,
              "--model", tiny["model"], "--max-batch-duration", "8", "--num-workers", "0", "--valid-interval", "0",
              "--save-interval", "0", "--valid-decode-utts", "2", "--valid-max-new-tokens", "4"]
    assert main(common + ["--num-epochs", "1"]) == 0
    s1 = json.load(open(os.path.join(exp, "train_summary.json")))
    assert main(common + ["--num-epochs", "2", "--resume"]) == 0
    s2 = json.load(open(os.path.join(exp, "train_summary.json")))
    assert s2["epoch"] == 2 and s2["global_step"] > s1["global_step"]
    assert os.path.exists(os.path.join(exp, "best", "model.safetensors"))

    lexp = str(tmp_path / "lora")
    assert main(["train", "--train-manifest", tiny["train"], "--exp-dir", lexp, "--model", tiny["model"],
                 "--num-epochs", "1", "--max-batch-duration", "8", "--num-workers", "0", "--lora",
                 "--lora-r", "4", "--save-interval", "0"]) == 0
    assert os.path.exists(os.path.join(lexp, "final", "model.safetensors"))  # merged at end
    adapter = os.path.join(lexp, "checkpoints")
    assert os.listdir(adapter)


def test_prepare(tiny, tmp_path):
    out = str(tmp_path / "prepared.tsv")
    assert main(["prepare", tiny["train"], "-o", out, "--max-duration", "1.5"]) == 0
    from cohere_transcribe_ara_cli.data import read_manifest

    segs = read_manifest(out, header="yes")
    assert segs and all(s.duration is not None and s.duration <= 1.5 for s in segs)


def test_two_process_decode_matches_single(tiny, tmp_path):
    """--gpus 2 relaunches with torchrun; on CPU this uses gloo. Output must equal single-process decoding."""
    import subprocess
    import sys

    one, two = str(tmp_path / "one"), str(tmp_path / "two")
    base = ["decode", tiny["valid"], "--model", tiny["model"], "--num-workers", "0", "--max-new-tokens", "6"]
    assert main(base + ["-o", one]) == 0
    r = subprocess.run([sys.executable, "-m", "cohere_transcribe_ara_cli", *base, "-o", two, "--gpus", "2",
                        "--max-batch-duration", "3"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert sorted(os.listdir(os.path.join(two, "parts"))) == [
        "hyp.rank0.tsv", "hyp.rank1.tsv", "stats.rank0.json", "stats.rank1.json"]
    read = lambda d: open(os.path.join(d, "hyp.tsv"), encoding="utf-8").read()  # noqa: E731
    assert read(one) == read(two)


def test_two_process_ddp_train(tiny, tmp_path):
    import subprocess
    import sys

    exp = str(tmp_path / "ddp")
    r = subprocess.run([sys.executable, "-m", "cohere_transcribe_ara_cli", "train", "--gpus", "2",
                        "--train-manifest", tiny["train"], "--valid-manifest", tiny["valid"], "--exp-dir", exp,
                        "--model", tiny["model"], "--num-epochs", "2", "--max-batch-duration", "6",
                        "--num-workers", "0", "--grad-accum", "2", "--spec-augment", "--valid-interval", "2",
                        "--save-interval", "2", "--valid-decode-utts", "4", "--valid-max-new-tokens", "4"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "processes=2" in r.stderr
    s = json.load(open(os.path.join(exp, "train_summary.json")))
    assert s["epoch"] == 2 and os.path.exists(os.path.join(exp, "final", "model.safetensors"))


def test_long_audio_is_chunked_and_reassembled(tiny, tmp_path):
    rng = np.random.default_rng(0)
    wav = str(tmp_path / "long.wav")
    sf.write(wav, (0.1 * rng.standard_normal(16000 * 70)).astype(np.float32), 16000)
    short = str(tmp_path / "short.wav")
    sf.write(short, (0.1 * rng.standard_normal(16000 * 2)).astype(np.float32), 16000)
    man = tmp_path / "long.tsv"
    man.write_text(f"long\t{wav}\tنص\nshort\t{short}\tنص\n", encoding="utf-8")

    from transformers import AutoProcessor

    from cohere_transcribe_ara_cli.data import DecodeCollator, load_audio

    fe = AutoProcessor.from_pretrained(tiny["model"]).feature_extractor
    batch = DecodeCollator(fe)([{"idx": 0, "audio": load_audio(wav)}, {"idx": 1, "audio": load_audio(short)}])
    chunks = [c for s, c in batch["audio_chunk_index"] if s == 0]
    assert len(chunks) >= 3 and batch["input_features"].shape[0] == len(batch["audio_chunk_index"])

    out = str(tmp_path / "dec")
    assert main(["decode", str(man), "-o", out, "--model", tiny["model"], "--num-workers", "0",
                 "--max-new-tokens", "5", "--tokens-per-second", "0"]) == 0
    lines = open(os.path.join(out, "hyp.tsv"), encoding="utf-8").read().splitlines()
    assert [ln.split("\t")[0] for ln in lines[1:]] == ["long", "short"]
    assert len(lines[1].split("\t")[2].split()) > len(lines[2].split("\t")[2].split())  # chunks re-joined
