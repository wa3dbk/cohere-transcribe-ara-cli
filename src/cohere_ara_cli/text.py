"""Text normalization used for scoring (and optionally for training targets).

Presets
-------
none           : only Unicode NFKC + whitespace collapsing (raw scoring)
basic          : none + casefold + punctuation/symbol removal
arabic         : basic + diacritics/tatweel removal, alef/yeh/kaf unification,
                 Arabic-Indic digits -> ASCII digits   (default for scoring)
arabic-strict  : arabic + ta marbuta -> heh, hamza-on-carrier -> carrier
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# Harakat, tanween, shadda, sukun, maddah/hamza marks, superscript alef, Quranic annotation marks.
_DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed\u08d3-\u08ff]"
)
_TATWEEL = "\u0640"
_ALEF_RE = re.compile("[\u0622\u0623\u0625\u0671\u0672\u0673]")  # آ أ إ ٱ ...
_WS_RE = re.compile(r"\s+")
_APOSTROPHES = "'\u2019\u2018\u02bc`"

_CHAR_MAP_ARABIC = {
    "\u0649": "\u064a",  # alef maksura -> yeh
    "\u06cc": "\u064a",  # farsi yeh -> yeh
    "\u06a9": "\u0643",  # keheh -> kaf
}
_CHAR_MAP_STRICT = {
    "\u0629": "\u0647",  # ta marbuta -> heh
    "\u0624": "\u0648",  # waw with hamza -> waw
    "\u0626": "\u064a",  # yeh with hamza -> yeh
}
_DIGITS = {ord(c): str(i) for i, c in enumerate("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")})


@dataclass(frozen=True)
class NormalizerConfig:
    casefold: bool = False
    remove_punct: bool = False
    remove_diacritics: bool = False
    remove_tatweel: bool = False
    unify_alef: bool = False
    unify_yeh_kaf: bool = False
    ascii_digits: bool = False
    strict_letters: bool = False


PRESETS: dict[str, NormalizerConfig] = {
    "none": NormalizerConfig(),
    "basic": NormalizerConfig(casefold=True, remove_punct=True),
    "arabic": NormalizerConfig(
        casefold=True,
        remove_punct=True,
        remove_diacritics=True,
        remove_tatweel=True,
        unify_alef=True,
        unify_yeh_kaf=True,
        ascii_digits=True,
    ),
    "arabic-strict": NormalizerConfig(
        casefold=True,
        remove_punct=True,
        remove_diacritics=True,
        remove_tatweel=True,
        unify_alef=True,
        unify_yeh_kaf=True,
        ascii_digits=True,
        strict_letters=True,
    ),
}


def _strip_punct(text: str) -> str:
    out = []
    for ch in text:
        if ch in _APOSTROPHES:
            continue  # "don't" -> "dont", avoids splitting words
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(text: str, cfg: NormalizerConfig) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    if cfg.remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    if cfg.remove_tatweel:
        text = text.replace(_TATWEEL, "")
    if cfg.unify_alef:
        text = _ALEF_RE.sub("\u0627", text)
    if cfg.unify_yeh_kaf:
        text = "".join(_CHAR_MAP_ARABIC.get(c, c) for c in text)
    if cfg.strict_letters:
        text = "".join(_CHAR_MAP_STRICT.get(c, c) for c in text)
    if cfg.ascii_digits:
        text = text.translate(_DIGITS)
    if cfg.casefold:
        text = text.casefold()
    if cfg.remove_punct:
        text = _strip_punct(text)
    return _WS_RE.sub(" ", text).strip()


def get_normalizer(name: str) -> Callable[[str], str]:
    """Return a callable normalizer for a preset name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown normalizer {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]
    return lambda s: normalize_text(s, cfg)
