"""Answer Language (GLOSSARY): ar for Arabic script, en for English, mixed only for Franco-Arabic."""
import pytest

from medrag.language import detect_language, refine


@pytest.mark.parametrize("message", [
    "ايه المادة الفعالة في بروفين؟", "فولتارين فيه مادة فعالة إيه؟", "هل Lasix بيعالج السكر؟", "ﺑﺮوﻓﻴﻦ",  # drug name in Latin: still ar
    "What is the active ingredient of Controloc? ما هي",
])
def test_arabic_script_is_ar(message):
    assert detect_language(message) == "ar"


@pytest.mark.parametrize("message", [
    "3andy so5oneya a5od eh", "el metformin mamnoo3 emta?", "eh el a3rad el gaanbeya bta3t Ventolin?",
    "Controloc da el maada el fa3ala beta3to eh?", "fe bedail tegareya le Brufen?", "a3mel kam mg Panadol fel yom?",
    "bdayel brufen eh?", "dawa el malaria eh", "mama waga3 sadr shedeed w 3aree2 bared", "ebny 5ad 10 7abat Zyrtec",
    "dawa lel daghT el 3aly",
])
def test_franco_is_mixed(message):
    assert detect_language(message) == "mixed"


@pytest.mark.parametrize("message", [
    "What is the active ingredient of Controloc?", "Is Zyrtec safe in pregnancy?", "How much caffeine is in Panadol Extra?",
    "Is Omega3 safe? Type2 diabetes, H2 blockers, vitamin B12 and D3", "Brufen 400 mg, 5mcg, 2nd line, 3rd dose, CYP3A4, PCSK9, SGLT2",
    "KB-03 side effects of Ventolin", "Which drug class is Zestril in?", "Can I take 2tablets or 3times a day for 5days?",
])
def test_english_with_drug_names_and_digits_is_en(message):
    assert detect_language(message) == "en"


def test_refine_needs_the_router_and_a_franco_mark_of_the_messages_own():
    assert detect_language("ezay astakhdem Panadol") == "en"  # one lexicon word alone is not enough
    assert refine("en", "ezay astakhdem Panadol", "mixed") == "mixed"  # one lexicon word + router
    assert refine("en", "What is Panadol", "mixed") == "en"  # the router alone never relabels English
    assert refine("ar", "ايه كتافلام", "mixed") == "ar"
    assert refine("en", "ezay astakhdem Panadol", None) == "en"
