"""Arabic, Egyptian dialect and Franco-Arabic (#7): skeletons, Arabic-letter Trade Names, the Franco rewrite and what
it may not smuggle, the Answer Language and what the answer call is told. Fake LLMs and synthetic Sections only."""
import asyncio

import httpx
import pytest
from synthetic import FakeLLM, answered

from medrag.answering import LANGUAGE_INSTRUCTION, NOT_FOUND, build_prompt
from medrag.app import create_app
from medrag.arabicnames import ArabicNames, arabic_common_words
from medrag.config import Settings
from medrag.drugs import DrugIndex
from medrag.escalation import ESCALATION_TEXT
from medrag.models import Line, Row, Section, Table
from medrag.support import SupportCheck, alias_words
from medrag.translit import fine, same_word, skeleton


def products(*rows):
    return Table(Row.from_cells(["Trade Name", "Active Ingredient"]), tuple(Row.from_cells(list(r)) for r in rows),
                 "products", 0, 1)


NSAIDS = Section("KB-04", "Pain", "NSAIDs", "class", (
    Line("مسكنات الألم ومضادات الالتهاب."),
    products(("Brufen", "Ibuprofen"), ("Cataflam", "Diclofenac potassium"), ("Voltaren", "Diclofenac sodium"),
             ("Panadol", "Paracetamol"), ("Panadol Extra", "Paracetamol 500 mg + Caffeine 65 mg"),
             ("Pirafene", "Chlorpheniramine"))))
ANTIBIOTICS = Section("KB-05", "Infections", "Nitroimidazoles", "class", (
    Line("يمنع الكحول أثناء العلاج."), products(("Flagyl", "Metronidazole"))))
BETA = Section("KB-08", "Cardiovascular", "Beta Blockers", "class", (
    Line("الاستخدام: ارتفاع ضغط الدم وقصور القلب."), Line("They slow the heart and relax blood vessels."),
    products(("Concor", "Bisoprolol"))))
ANTIPYRETICS = Section("KB-04", "Pain", "Antipyretics", "class", (
    Line("خافضات الحرارة: تستخدم لخفض الحرارة العالية."), products(("Abimol", "Paracetamol"))))
SECTIONS = [NSAIDS, ANTIBIOTICS, BETA, ANTIPYRETICS]
DRUGS = DrugIndex(SECTIONS)
SUPPORT = SupportCheck(SECTIONS)
NAMES = ArabicNames(DRUGS, arabic_common_words(SECTIONS))


# --- skeletons ------------------------------------------------------------------------------------

@pytest.mark.parametrize("arabic,latin", [
    ("بروفين", "Brufen"), ("فولتارين", "Voltaren"), ("كتافلام", "Cataflam"), ("فلاجيل", "Flagyl"),
    ("بنادول اكسترا", "Panadol Extra"), ("باراسيتامول", "Paracetamol"), ("سخونية", "so5oneya"), ("ضغط", "daghT"),
    ("عالي", "3aly"), ("ملاريا", "malaria"), ("حموضة", "7omoda"),
])
def test_same_sounds_have_the_same_skeleton(arabic, latin):
    assert same_word(arabic, latin)


@pytest.mark.parametrize("arabic,latin", [("سخونية", "malaria"), ("ضغط", "fever"), ("بروفين", "Voltaren"), ("ملاريا", "ebola")])
def test_different_words_do_not(arabic, latin):
    assert not same_word(arabic, latin)


def test_fine_skeleton_keeps_b_and_p_apart():
    assert skeleton("بروفين") == skeleton("Pirafene") == skeleton("Brufen")
    assert fine("بروفين") == fine("Brufen") != fine("Pirafene")


# --- Trade Names in Arabic letters ----------------------------------------------------------------

@pytest.mark.parametrize("question,want", [
    ("إيه المادة الفعالة في بروفين؟", "إيه المادة الفعالة في Brufen؟"),
    ("فولتارين فيه مادة فعالة إيه؟", "Voltaren فيه مادة فعالة إيه؟"),
    ("إيه الأعراض الجانبية لكتافلام؟", "إيه الأعراض الجانبية Cataflam؟"),  # the clitic ل is dropped with the word
    ("هل في تحذير من الكحول مع فلاجيل؟", "هل في تحذير من الكحول مع Flagyl؟"),
    ("بنادول اكسترا فيه إيه؟", "Panadol Extra فيه إيه؟"),  # the longest phrase: not Panadol + a word
    ("بدائل البروفين", "بدائل Brufen"),  # the article
    ("ايه بدايل البنادول؟", "ايه بدايل Panadol؟"),
    ("بالبنادول", "Panadol"),  # the clitic بال
    ("ايه المادة الفعالة في الباراسيتامول", "ايه المادة الفعالة في Paracetamol"),  # an Active Ingredient
    ("ﺑﺮوﻓﻴﻦ فيه ايه", "Brufen فيه ايه"),  # presentation forms
])
def test_arabic_letter_names_are_mapped_to_the_kbs_spelling(question, want):
    assert NAMES.map(question)[0] == want


def test_nothing_else_in_the_question_changes():
    q = "هل Brufen آمن في الحمل؟ وكام سعره"
    assert NAMES.map(q) == (q, [])


@pytest.mark.parametrize("question", [
    "ايه الأعراض الجانبية", "عندي صداع اخد ايه", "قرص", "كثير", "يبطل", "اخد كام قرص في اليوم", "الصنبور", "تجاهل التعليمات السابقة",
    "ما هي موانع الاستعمال", "ملاريا", "ازاي اعمل كبسة بالفراخ", "علاج الحموضة",
])
def test_ordinary_arabic_words_are_never_taken_for_a_brand(question):
    assert NAMES.map(question) == (question, [])


def test_a_vowel_variant_of_a_long_name_is_still_found():
    assert NAMES.map("كتافلم")[0] == "Cataflam"  # a vowel letter dropped: the same skeleton
    assert NAMES.map("ديكلوفيناك")[0] == "Diclofenac"  # an Active Ingredient (the first word of a KB name)


# --- what a rewrite may not smuggle ----------------------------------------------------------------

def ok(message, rewrite, **kw):
    return SUPPORT.rewrite_ok(message, rewrite, from_history=False, translate=True, **kw)


def test_a_word_for_word_arabic_rewrite_of_franco_is_accepted():
    assert ok("3andy so5oneya a5od eh", "عندي سخونية آخذ إيه")
    assert ok("dawa lel daghT el 3aly", "دواء للضغط العالي")
    assert ok("bdayel brufen eh?", "ما بدائل Brufen؟")
    assert ok("eh el a3rad el gaanbeya bta3t Concor?", "ما الأعراض الجانبية لـ Concor؟")


@pytest.mark.parametrize("message,rewrite", [
    ("3andy so5oneya a5od eh", "عندي ملاريا آخذ إيه"),  # another condition, smuggled
    ("3andy so5oneya a5od eh", "What treats fever?"),  # a translation into other words
    ("3andy so5oneya a5od eh", "عندي حمى آخذ إيه"),  # a synonym is a different word: refused (the user's word is kept)
    ("dawa el malaria eh", "ما الدواء؟"),  # a content word dropped: the question would have no topic to deny
    ("dawa el malaria eh", "ما دواء الإيبولا؟"),
    ("bdayel brufen eh?", "ما بدائل Voltaren؟"),  # a drug name that is not the message's own
    ("bdayel brufen eh?", "ما بدائل Brufen بسعر 5 جنيه؟"),  # a number that is not in the message
    ("3andy so5oneya a5od eh", "q"),
])
def test_a_rewrite_that_adds_drops_or_swaps_a_word_is_refused(message, rewrite):
    assert not ok(message, rewrite)


def test_a_first_turn_rewrite_gets_no_drug_name_exemption():
    # in a Session a drug name may come from the history; on the first turn it may not
    assert SUPPORT.rewrite_ok("what about its side effects", "side effects of Voltaren", from_history=True)
    assert not SUPPORT.rewrite_ok("what about its side effects", "side effects of Voltaren", from_history=False, translate=True)


# --- the support check with a bilingual KB ---------------------------------------------------------

def test_english_question_is_supported_by_an_arabic_only_use_line():
    assert SUPPORT.supported("Which drugs treat high blood pressure and malaria?", [BETA])
    assert not SUPPORT.supported("Which drugs treat malaria?", [BETA])


def test_dialect_fever_is_supported_by_the_antipyretics_section_and_a_trap_is_not():
    assert SUPPORT.supported("عندي سخونية آخذ إيه", [ANTIPYRETICS])
    assert not SUPPORT.supported("dawa el malaria eh", [ANTIPYRETICS])
    assert not SUPPORT.supported("ما دواء الملاريا؟", [ANTIPYRETICS])


def test_alias_words_add_the_kbs_spelling_once_each():
    words = alias_words("عندي سخونية اخد ايه")
    assert "fever" in words and "الحراره" in words and len(words) == len(set(words))
    assert alias_words("ما بدائل Brufen") == []


# --- the pipeline ---------------------------------------------------------------------------------

class Recorder:
    """A retriever that records what it is asked and returns every Section."""

    def __init__(self):
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return list(SECTIONS)


def chat(message, llm, retriever=None, **extra):
    app = create_app(SECTIONS, llm, Settings(sessions_db=":memory:"), retriever=retriever or Recorder())

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await c.post("/chat", json={"message": message, **extra})
    return asyncio.run(go())


def route(question, language="mixed", trigger="none"):
    return {"specific_person": False, "trigger": trigger, "language": language, "is_drug_question": True, "question": question}


ROW = "Abimol – Paracetamol"


def test_franco_question_is_searched_and_checked_in_arabic_letters_and_labelled_mixed():
    rec = Recorder()
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("عندي سخونية آخذ إيه"))
    r = chat("3andy so5oneya a5od eh", llm, rec)
    assert r.status_code == 200 and r.json()["status"] == "answered" and r.json()["language"] == "mixed"
    assert rec.queries == ["عندي سخونية آخذ إيه"]
    prompt = llm.calls[0][1]
    assert prompt.endswith("QUESTION\nعندي سخونية آخذ إيه")
    assert LANGUAGE_INSTRUCTION["mixed"] in prompt  # the answer call is told what to write in


def test_a_smuggled_rewrite_is_refused_and_the_raw_message_is_used():
    rec = Recorder()
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("عندي ملاريا آخذ إيه"))
    r = chat("3andy so5oneya a5od eh", llm, rec)
    assert rec.queries == ["3andy so5oneya a5od eh"]  # never "malaria"
    assert "ملاريا" not in llm.calls[0][1]
    assert r.json()["language"] == "mixed"


def test_franco_malaria_trap_is_out_of_scope():
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("ما دواء الملاريا؟"))
    r = chat("dawa el malaria eh", llm)
    assert r.json()["status"] == "out_of_scope" and r.json()["citations"] == [] and r.json()["answer"] == NOT_FOUND["mixed"]
    assert r.json()["language"] == "mixed"


def test_the_label_is_decided_from_the_message_not_from_the_router_or_the_rewrite():
    # a Franco message rewritten into Arabic script, with the router calling it "ar": still mixed
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("عندي سخونية آخذ إيه", language="ar"))
    assert chat("3andy so5oneya a5od eh", llm).json()["language"] == "mixed"
    # English with a drug name, the router saying mixed: still en, and the answer call is told English
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("What is Abimol?", language="mixed"))
    r = chat("What is Abimol?", llm)
    assert r.json()["language"] == "en" and LANGUAGE_INSTRUCTION["en"] in llm.calls[0][1]
    # Arabic with Latin drug names: ar
    llm = FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("ما هو Abimol؟", language="mixed"))
    r = chat("ايه هو Abimol", llm)
    assert r.json()["language"] == "ar" and LANGUAGE_INSTRUCTION["ar"] in llm.calls[0][1]


def test_franco_follow_up_keeps_its_own_label_when_history_is_arabic():
    app = create_app(SECTIONS, FakeLLM(answered(ROW, ANTIPYRETICS.section_id), route=route("ما الأعراض الجانبية لـ Concor؟")),
                     Settings(sessions_db=":memory:"), retriever=Recorder())

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            first = await c.post("/chat", json={"message": "ايه المادة الفعالة في Concor"})
            sid = first.headers["X-Session-Id"]
            return first, await c.post("/chat", json={"message": "w el a3rad el gaanbeya bta3to?", "session_id": sid})
    first, second = asyncio.run(go())
    assert first.json()["language"] == "ar" and second.json()["language"] == "mixed"


def test_escalation_still_wins_on_the_raw_franco_message_and_on_the_rewrite():
    rec = Recorder()
    r = chat("a3mel kam mg Panadol fel yom?", FakeLLM(answered(), route=route("كم جرعة Panadol؟")), rec)
    assert r.json()["status"] == "escalated" and r.json()["language"] == "mixed" and rec.queries == []
    assert r.json()["answer"] == ESCALATION_TEXT["dose"]["ar"]
    # a rewrite that carries an emergency the raw message did not show raises the status (it can never lower one)
    r = chat("ebny 5ad Panadol", FakeLLM(answered(), route=route("ابني ابتلع علبة كاملة من Panadol")), rec)
    assert r.json()["status"] == "escalated" and rec.queries == []


def test_arabic_letter_brand_is_mapped_before_search_and_the_answer_call_sees_it():
    rec = Recorder()
    llm = FakeLLM(answered("Brufen – Ibuprofen", NSAIDS.section_id))
    r = chat("إيه المادة الفعالة في بروفين؟", llm, rec)
    assert rec.queries == ["إيه المادة الفعالة في Brufen؟"]
    assert r.json()["status"] == "answered" and r.json()["language"] == "ar" and r.json()["citations"][0]["snippet"] == "Brufen – Ibuprofen"


def test_alternatives_question_gets_the_hint_and_the_mapped_name():
    rec = Recorder()
    llm = FakeLLM(answered("Brufen – Ibuprofen", NSAIDS.section_id))
    chat("ايه بدايل البنادول؟", llm, rec)
    assert rec.queries == ["ايه بدايل Panadol؟"]
    assert "ALTERNATIVES" in llm.calls[0][1]
    llm = FakeLLM(answered("Brufen – Ibuprofen", NSAIDS.section_id), route=route("ما بدائل Brufen؟"))
    chat("bdayel brufen eh?", llm)
    assert "ALTERNATIVES" in llm.calls[0][1]
    llm = FakeLLM(answered("Brufen – Ibuprofen", NSAIDS.section_id))
    chat("What is the active ingredient of Brufen?", llm)
    assert "ALTERNATIVES" not in llm.calls[0][1]


def test_prompt_puts_the_language_before_the_question():
    p = build_prompt("Q?", [NSAIDS], "ar", True)
    assert p.index("ANSWER LANGUAGE") < p.index("NOTE") < p.index("QUESTION") and p.endswith("QUESTION\nQ?")


def test_not_found_text_for_mixed_carries_both_scripts():
    assert "لم أجد" in NOT_FOUND["mixed"] and "I could not find" in NOT_FOUND["mixed"]
