"""The shared normaliser (#7): every Arabic variant folds to one form, questions and Sections alike."""
import pytest

from medrag.rules import normalize as rules_normalize
from medrag.textnorm import has_arabic, normalize


@pytest.mark.parametrize("raw,want", [
    ("أحمد إبراهيم آمنة ٱلله", "احمد ابراهيم امنه الله"),  # alef variants
    ("مستشفى على", "مستشفي علي"),  # alef maqsura -> yaa
    ("حموضة الحمل", "حموضه الحمل"),  # taa marbuta -> haa
    ("مؤشر شيء مسئول", "موشر شيء مسيول"),  # hamza on waw / yaa
    ("كَتَافْلَام", "كتافلام"),  # harakat
    ("بروفيـــن", "بروفين"),  # tatweel
    ("جرعة ٥٠٠ مجم و ۱۰ حبة", "جرعه 500 مجم و 10 حبه"),  # Arabic-Indic and Persian digits
    ("ﺑﺮوﻓﻴﻦ", "بروفين"),  # presentation forms B (U+FE70-FEFF), as extracted from a PDF
    ("ﷲ", "الله"),  # presentation form A (U+FB50-FDFF): the ligature of the name
    ("ﻻ", "لا"),  # lam-alef ligature
    ("کتاب فارسی", "كتاب فارسي"),  # Persian kaf and yeh
    ("  Brufen   400\tMG ", "brufen 400 mg"),  # case and whitespace
    ("it’s `x`", "it's 'x'"),  # quotes
])
def test_normalize_folds_variant(raw, want):
    assert normalize(raw) == want


def test_normalize_is_idempotent_and_shared_with_the_rules():
    s = "أَدْوِيَةُ الضَّغْطِ ﺑﺮوﻓﻴﻦ ٢٣"
    assert normalize(normalize(s)) == normalize(s)
    assert rules_normalize is normalize


def test_has_arabic_sees_presentation_forms_and_not_latin():
    assert has_arabic("ﺑﺮوﻓﻴﻦ")  # before NFKC: the regex itself covers U+FB50-FDFF and U+FE70-FEFF
    assert has_arabic("hello ب")
    assert not has_arabic("Brufen 400 mg 3andy")
