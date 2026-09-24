import math

from cohere_transcribe_ara_cli.metrics import align_words, count_edits, format_alignment, score
from cohere_transcribe_ara_cli.text import get_normalizer


def test_counts():
    c = count_edits("a b c d".split(), "a x c d e".split())
    assert (c.sub, c.dele, c.ins, c.ref) == (1, 0, 1, 4)
    assert math.isclose(c.rate, 0.5)


def test_empty_sides():
    assert count_edits([], ["a", "b"]).ins == 2
    assert count_edits(["a", "b"], []).dele == 2


def test_alignment_roundtrip():
    al = align_words("a b c d".split(), "a c d e".split())  # unambiguous: delete b, insert e
    assert [o for o, _, _ in al] == ["C", "D", "C", "C", "I"]
    ref, hyp, ops = format_alignment(al)
    assert "***" in ref and "***" in hyp and "D" in ops and "I" in ops


def test_score_report():
    items = [("u1", "مرحبا بكم", "مرحبا بكم"), ("u2", "شكرا جزيلا", "شكرا"), ("u3", "", "زائد")]
    rep, utts = score(items, get_normalizer("arabic"), "arabic")
    assert rep.words.ref == 4 and rep.words.dele == 1 and rep.words.ins == 1
    assert math.isclose(rep.wer, 0.5)
    assert rep.num_sentence_errors == 2 and len(utts) == 3
    assert rep.to_dict()["wer"] == 50.0
