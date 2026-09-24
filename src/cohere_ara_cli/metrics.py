"""WER / CER scoring with per-utterance alignments (rapidfuzz backend, C++ speed)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

from rapidfuzz.distance import Levenshtein


@dataclass
class EditCounts:
    ref: int = 0
    hyp: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        if self.ref == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.ref

    def __iadd__(self, other: EditCounts) -> EditCounts:
        self.ref += other.ref
        self.hyp += other.hyp
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        return self


def count_edits(ref: Sequence, hyp: Sequence) -> EditCounts:
    """Levenshtein S/D/I counts between two sequences (strings or token lists)."""
    c = EditCounts(ref=len(ref), hyp=len(hyp))
    if not ref:
        c.ins = len(hyp)
        return c
    if not hyp:
        c.dele = len(ref)
        return c
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "replace":
            c.sub += 1
        elif op.tag == "delete":
            c.dele += 1
        elif op.tag == "insert":
            c.ins += 1
    return c


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Return an alignment as (op, ref_word, hyp_word); op in {C, S, D, I}."""
    if not ref:
        return [("I", None, h) for h in hyp]
    if not hyp:
        return [("D", r, None) for r in ref]
    out: list[tuple[str, str | None, str | None]] = []
    for op in Levenshtein.opcodes(ref, hyp):
        r = ref[op.src_start : op.src_end]
        h = hyp[op.dest_start : op.dest_end]
        if op.tag == "equal":
            out.extend(("C", a, b) for a, b in zip(r, h))
        elif op.tag == "replace":
            n = max(len(r), len(h))
            for i in range(n):
                a = r[i] if i < len(r) else None
                b = h[i] if i < len(h) else None
                out.append(("S" if a is not None and b is not None else ("D" if b is None else "I"), a, b))
        elif op.tag == "delete":
            out.extend(("D", a, None) for a in r)
        elif op.tag == "insert":
            out.extend(("I", None, b) for b in h)
    return out


def format_alignment(alignment: list[tuple[str, str | None, str | None]]) -> tuple[str, str, str]:
    """sclite-like REF/HYP/OPS lines (*** marks insertions/deletions)."""
    ref_toks, hyp_toks, ops = [], [], []
    for op, r, h in alignment:
        r_s = r if r is not None else "*" * max(3, len(h or ""))
        h_s = h if h is not None else "*" * max(3, len(r or ""))
        w = max(len(r_s), len(h_s))
        ref_toks.append(r_s.ljust(w))
        hyp_toks.append(h_s.ljust(w))
        ops.append(("" if op == "C" else op).ljust(w))
    return " ".join(ref_toks), " ".join(hyp_toks), " ".join(ops)


@dataclass
class UttScore:
    id: str
    ref: str
    hyp: str
    words: EditCounts
    chars: EditCounts


@dataclass
class ScoreReport:
    normalizer: str
    num_utts: int = 0
    num_sentence_errors: int = 0
    words: EditCounts = field(default_factory=EditCounts)
    chars: EditCounts = field(default_factory=EditCounts)
    top_substitutions: list = field(default_factory=list)
    top_deletions: list = field(default_factory=list)
    top_insertions: list = field(default_factory=list)

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def cer(self) -> float:
        return self.chars.rate

    def to_dict(self) -> dict:
        return {
            "normalizer": self.normalizer,
            "num_utts": self.num_utts,
            "wer": round(100 * self.wer, 3),
            "cer": round(100 * self.cer, 3),
            "ser": round(100 * self.num_sentence_errors / max(1, self.num_utts), 3),
            "words": asdict(self.words),
            "chars": asdict(self.chars),
            "top_substitutions": self.top_substitutions,
            "top_deletions": self.top_deletions,
            "top_insertions": self.top_insertions,
        }

    def summary_line(self) -> str:
        w = self.words
        return (
            f"[{self.normalizer}] WER {100 * self.wer:.2f}% "
            f"[ {w.errors} / {w.ref}, {w.ins} ins, {w.dele} del, {w.sub} sub ] "
            f"CER {100 * self.cer:.2f}%  SER {100 * self.num_sentence_errors / max(1, self.num_utts):.2f}% "
            f"({self.num_utts} utts)"
        )


def score(
    items: Iterable[tuple[str, str, str]],
    normalizer: Callable[[str], str],
    normalizer_name: str,
    top_k: int = 20,
    keep_utts: bool = True,
) -> tuple[ScoreReport, list[UttScore]]:
    """Score (id, ref, hyp) triples. CER is computed on normalized strings including spaces."""
    report = ScoreReport(normalizer=normalizer_name)
    subs: Counter = Counter()
    dels: Counter = Counter()
    inss: Counter = Counter()
    utts: list[UttScore] = []
    for uid, ref, hyp in items:
        r = normalizer(ref or "")
        h = normalizer(hyp or "")
        rw, hw = r.split(), h.split()
        wc = count_edits(rw, hw)
        cc = count_edits(r, h)
        report.num_utts += 1
        report.words += wc
        report.chars += cc
        if wc.errors:
            report.num_sentence_errors += 1
            for op, a, b in align_words(rw, hw):
                if op == "S":
                    subs[f"{a} -> {b}"] += 1
                elif op == "D":
                    dels[a] += 1
                elif op == "I":
                    inss[b] += 1
        if keep_utts:
            utts.append(UttScore(uid, r, h, wc, cc))
    report.top_substitutions = subs.most_common(top_k)
    report.top_deletions = dels.most_common(top_k)
    report.top_insertions = inss.most_common(top_k)
    return report, utts


def write_report(path: str, reports: list[ScoreReport]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({r.normalizer: r.to_dict() for r in reports}, f, ensure_ascii=False, indent=2)


def write_details(path: str, utts: list[UttScore]) -> None:
    """Per-utterance alignment file, worst utterances first."""
    order = sorted(utts, key=lambda u: (-u.words.errors, u.id))
    with open(path, "w", encoding="utf-8") as f:
        for u in order:
            ref_l, hyp_l, ops_l = format_alignment(align_words(u.ref.split(), u.hyp.split()))
            w = u.words
            f.write(
                f"id: {u.id}\n"
                f"Scores: (#C #S #D #I) {w.ref - w.sub - w.dele} {w.sub} {w.dele} {w.ins}  "
                f"WER {100 * w.rate if w.ref else 0.0:.1f}%\n"
                f"REF:  {ref_l}\nHYP:  {hyp_l}\nOPS:  {ops_l}\n\n"
            )
