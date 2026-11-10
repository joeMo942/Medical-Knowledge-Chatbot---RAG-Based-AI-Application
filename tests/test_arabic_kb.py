"""#7 on the real KB (skipped without the .docx files): ordinary Arabic words are never taken for a drug, the Arabic-letter
brands still are, and a Franco rewrite cannot swap one condition for another."""
import pytest

from medrag.arabicnames import ArabicNames, arabic_common_words
from medrag.drugs import DrugIndex
from medrag.support import SupportCheck


@pytest.fixture(scope="module")
def kb(real_kb):
    sections = [s for d in real_kb.values() for s in d.sections]
    return ArabicNames(DrugIndex(sections), arabic_common_words(sections)), SupportCheck(sections)


NOT_DRUGS = [
    "سرطان", "السرطان", "للسرطان", "والسرطان", "بالسرطان", "لسرطان الثدي", "التسمم", "تسمم", "الناسور", "السكر", "الضغط", "الربو",
    "الملاريا", "السل", "الصرع", "القلب", "الكلى", "الكبد", "الصدفية", "الإيدز", "الايدز", "الدخان", "الدكان", "البيطري", "السبيل",
    "النسر", "كمبيوتر", "فرامل", "فلوموكس", "الكوليرا", "الجدري", "الجذام", "الزهري", "الحصبة", "الدرن", "الجلطة", "الزهايمر",
    "ايه علاج السرطان؟", "ادوية السرطان", "علاج الناسور", "ايه علاج التسمم", "عندي سرطان", "مريض سرطان ياخد ايه", "اضرار الدخان",
]


@pytest.mark.kb
@pytest.mark.parametrize("text", NOT_DRUGS)
def test_ordinary_arabic_words_map_to_nothing(kb, text):
    assert kb[0].map(text) == (text, [])


@pytest.mark.kb
@pytest.mark.parametrize("text,name", [
    ("إيه المادة الفعالة في بروفين؟", "Brufen"), ("فولتارين فيه مادة فعالة إيه؟", "Voltaren"), ("إيه الأعراض الجانبية لكتافلام؟", "Cataflam"),
    ("هل في تحذير من الكحول مع فلاجيل؟", "Flagyl"), ("بنادول اكسترا فيه إيه؟", "Panadol Extra"), ("بدائل البروفين", "Brufen"),
    ("ايه بدايل البنادول؟", "Panadol"), ("بالبنادول", "Panadol"), ("لكتافلام", "Cataflam"), ("بيرافين فيه ايه؟", "Pirafene"),
])
def test_arabic_letter_brands_still_map(kb, text, name):
    assert name in kb[0].map(text)[0]


@pytest.mark.kb
@pytest.mark.parametrize("message,rewrite", [
    ("dawa el ebola eh", "ما دواء البول؟"),
    ("dawa lel kalb eh", "ما دواء للقلب؟"),  # kalb is a dog; القلب is the heart
    ("dawa lel zohry eh", "ما دواء للظهر؟"),  # zohry is syphilis; الظهر is the back
])
def test_a_condition_swapped_for_another_is_refused(kb, message, rewrite):
    assert kb[1].translated(message, rewrite) is None
