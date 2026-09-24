from cohere_transcribe_ara_cli.text import get_normalizer


def test_none_only_whitespace():
    n = get_normalizer("none")
    assert n("  مرحبا   بكم ") == "مرحبا بكم"


def test_arabic_normalizer():
    n = get_normalizer("arabic")
    assert n("إِنَّ الأَمْرَ، سَهْلٌ!") == "ان الامر سهل"
    assert n("مستشفى") == "مستشفي"          # alef maksura -> yeh
    assert n("كتـــاب") == "كتاب"            # tatweel
    assert n("عام ٢٠٢٦") == "عام 2026"       # Arabic-Indic digits
    assert n("Hello, World؟") == "hello world"
    assert n("don't") == "dont"
    assert n("مدرسة") == "مدرسة"             # ta marbuta kept in 'arabic'


def test_arabic_strict():
    n = get_normalizer("arabic-strict")
    assert n("مدرسة مسؤول") == "مدرسه مسوول"


def test_presentation_forms_nfkc():
    assert get_normalizer("arabic")("\ufefb") == "لا"
