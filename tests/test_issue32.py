"""Issue 32: self-care questions, typo correction with "Did you mean", end to end with a fake LLM and synthetic Sections."""
import asyncio

import httpx
import pytest
from synthetic import NSAIDS, SECTIONS, FakeLLM, answered

from medrag.answering import NOT_FOUND
from medrag.app import create_app
from medrag.config import Settings
from medrag.gate import LLMGate
from medrag.models import Line, Section
from medrag.pipeline import decide
from medrag.selfcare import pharmacist_note
from medrag.sessions import Turn
from medrag.typos import TypoCorrector, damerau, did_you_mean

NOTE_EN = pharmacist_note("en")
NOTE_AR = pharmacist_note("ar")


def person(trigger, **extra):
    return {"specific_person": True, "trigger": trigger, "language": "en", "is_drug_question": True,
            "question": "q", **extra}


def post(llm, message, sections=SECTIONS):
    app = create_app(sections, llm, Settings())

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return (await c.post("/chat", json={"message": message})).json()

    return asyncio.run(go())


# ------------------------------------------------------------------------- self-care, end to end

def test_a_self_care_question_the_router_calls_a_treatment_decision_is_answered_with_the_pharmacist_note():
    llm = FakeLLM(reply=answered(), route=person("treatment_decision"))
    body = post(llm, "I have pain, what can I take?")
    assert body["status"] == "answered" and body["citations"]
    assert body["answer"].endswith("\n\n" + NOTE_EN)
    assert body["answer"].startswith("Diclofenac.")
    assert llm.calls and "NOTE" in llm.calls[0][1] and "own everyday symptom" in llm.calls[0][1]


def test_the_note_is_arabic_for_an_arabic_question():
    kb = Section("KB-03", "Pain", "Simple Analgesics", "class", (Line("الاستخدام: علاج الصداع والألم الخفيف."),))
    llm = FakeLLM(reply=answered(section_id=kb.section_id, snippet="الاستخدام: علاج الصداع والألم الخفيف."),
                  route={**person("treatment_decision"), "language": "ar"})
    body = post(llm, "عندي صداع اخد ايه", [kb])
    assert body["status"] == "answered" and body["language"] == "ar"
    assert body["answer"].endswith("\n\n" + NOTE_AR)


@pytest.mark.parametrize("message", [
    "I'm pregnant and have pain, what can I take?",
    "I have pain, I am 12 years old, what can I take?",
    "I'm diabetic and have pain, what can I take?",
    "I take warfarin and have pain, what can I take?",
    "I have severe pain, what can I take?",
    "I have had pain for 3 days, what can I take?",
    "My wife has pain, what can she take?",
    "I have pain, what can I give my son?",
])
def test_a_qualified_or_other_person_question_stays_escalated(message):
    llm = FakeLLM(reply=answered(), route=person("treatment_decision"))
    body = post(llm, message)
    assert body["status"] == "escalated" and body["citations"] == []
    assert not llm.calls  # decided before the answer call


@pytest.mark.parametrize("trigger", ["emergency", "dose", "diagnosis"])
def test_the_self_care_rule_never_lowers_emergency_dose_or_diagnosis(trigger):
    llm = FakeLLM(reply=answered(), route=person(trigger))
    body = post(llm, "I have pain, what can I take?")
    assert body["status"] == "escalated"


@pytest.mark.parametrize("message", ["I have pain, how many mg can I take?", "I have chest pain, what can I take?"])
def test_the_rules_still_escalate_a_dose_or_an_emergency_even_if_the_router_says_none(message):
    llm = FakeLLM(reply=answered(), route=person("none"))
    assert post(llm, message)["status"] == "escalated"


def test_decide_lowers_only_treatment_decision():
    def trig(message, t):
        return asyncio.run(decide(message, LLMGate(FakeLLM(route=person(t)))))

    d = trig("what can i take if i have pain", "treatment_decision")
    assert d.trigger is None and d.self_care
    d = trig("what can i take if i have pain", "diagnosis")
    assert d.trigger == "diagnosis"
    d = trig("what can i take if i have pain while pregnant", "treatment_decision")
    assert d.trigger == "treatment_decision" and not d.self_care


def test_a_qualifier_in_an_earlier_turn_keeps_the_escalation():
    llm = FakeLLM(route=person("treatment_decision"))
    history = [Turn("my son has a fever", "ok", "escalated")]
    d = asyncio.run(decide("what can i take for pain", LLMGate(llm), history=history))
    assert d.trigger == "treatment_decision" and not d.self_care


def test_when_the_routing_call_fails_a_backstop_treatment_decision_is_ignored_for_self_care_only():
    backstop = {**answered(), "specific_person": True, "trigger": "treatment_decision"}
    llm = FakeLLM(reply=backstop, route_exc=RuntimeError("down"))
    body = post(llm, "I have pain, what can I take?")
    assert body["status"] == "answered" and body["answer"].endswith(NOTE_EN)
    llm = FakeLLM(reply=backstop, route_exc=RuntimeError("down"))
    assert post(llm, "Can I give Cataflam to my son for pain?")["status"] == "escalated"


def test_the_not_found_text_gets_no_prefix_and_no_note():
    llm = FakeLLM(reply={"answer": "", "status": "out_of_scope", "citations": []}, route=person("none"))
    body = post(llm, "I have malaria, what can I take?")
    assert body["status"] == "out_of_scope" and body["answer"] == NOT_FOUND["en"]


# ------------------------------------------------------------------------------------ typo correction

KB = Section("KB-03", "Pain", "Simple Analgesics", "class",
             (Line("Used for headache, fever and mild pain. Avoid if pregnant."),
              Line("Ibuprofen and Brufen are NSAIDs. Pirafene is a brand."),
              Line("Hypertension is high blood pressure.")))
CORRECTOR = TypoCorrector([KB])


@pytest.mark.parametrize("word,expected", [("headace", "headache"), ("hedache", "headache"), ("Brufin", "Brufen"),
                                           ("pregnnat", "pregnant"), ("Ibuprofn", "Ibuprofen"), ("fevre", "fever")])
def test_a_word_one_edit_from_exactly_one_kb_word_is_corrected(word, expected):
    assert CORRECTOR.correct_word(word) == expected


def test_a_transposition_is_one_edit_and_two_edits_need_nine_letters():
    assert damerau("fevre", "fever", 1) == 1
    assert damerau("headace", "headache", 1) == 1
    assert damerau("abcdef", "abcxyf", 1) == 2
    assert CORRECTOR.correct_word("hypertnsion") == "Hypertension"
    assert CORRECTOR.correct_word("hypertnsoin") == "Hypertension"  # two edits, 11 letters
    assert CORRECTOR.correct_word("hedache") == "headache"  # one edit
    assert CORRECTOR.correct_word("hedachs") is None  # two edits, 7 letters


def test_an_ambiguous_word_is_never_corrected():
    sec = Section("KB-03", "Pain", "X", "class", (Line("Brufen and Brufan are brands."),))
    assert TypoCorrector([sec]).correct_word("Brufin") is None


def test_a_known_word_a_short_word_a_number_and_another_form_of_a_kb_word_are_left_alone():
    assert CORRECTOR.correct_word("Brufen") is None  # a KB drug is never turned into another one
    assert CORRECTOR.correct_word("fevr") is None  # four letters
    assert CORRECTOR.correct_word("12345") is None
    assert CORRECTOR.correct_word("headaches") is None  # an inflection, not a typo
    assert CORRECTOR.correct_word("barely") is None  # an everyday English word
    assert CORRECTOR.correct_word("3andy") is None and CORRECTOR.correct_word("so5oneya") is None
    assert CORRECTOR.correct_word("malaria") is None  # nothing in the KB is near it


def test_arabic_script_and_franco_arabic_are_never_corrected():
    assert CORRECTOR.correct("عندي صدااع اخد ايه").changes == ()
    assert CORRECTOR.correct("I have headace", "mixed").changes == ()
    assert CORRECTOR.correct("I have headace", "en").text == "I have headache"


def test_the_correction_replaces_the_word_and_keeps_the_rest():
    r = CORRECTOR.correct("what i can take if i have headace, Brufin?")
    assert r.text == "what i can take if i have headache, Brufen?"
    assert [(c.original, c.corrected) for c in r.changes] == [("headace", "headache"), ("Brufin", "Brufen")]


def test_did_you_mean_text():
    assert did_you_mean(["headache"], "en") == "Did you mean: headache?"
    assert did_you_mean(["headache", "headache", "Brufen"], "ar") == "هل تقصد: headache, Brufen؟"
    assert did_you_mean(["headache"], "mixed").startswith("هل تقصد")


# ----------------------------------------------------------------- "Did you mean" through the pipeline

def test_an_answered_reply_to_a_corrected_question_starts_with_did_you_mean():
    llm = FakeLLM(reply=answered(), route=person("none"))
    body = post(llm, "What is the active ingredient of Voltarin?")
    assert body["status"] == "answered"
    assert body["answer"] == "Did you mean: Voltaren?\nDiclofenac."
    assert "Voltaren" in llm.calls[0][1] and "Voltarin" not in llm.calls[0][1]


def test_an_answered_self_care_reply_to_a_corrected_question_has_the_prefix_and_the_note():
    llm = FakeLLM(reply=answered(), route=person("treatment_decision"))
    body = post(llm, "what i can take if i have inflamation")
    assert body["status"] == "answered"
    assert body["answer"] == f"Did you mean: inflammation?\nDiclofenac.\n\n{NOTE_EN}"


def test_the_rules_also_run_on_the_corrected_question():
    # "pregnnat" is a typo of a KB word: the qualifier only shows once it is corrected
    kb = Section("KB-03", "Pain", "Simple Analgesics", "class", (Line("Used for pain. Avoid if pregnant."),))
    llm = FakeLLM(reply=answered(section_id=kb.section_id, snippet="Used for pain. Avoid if pregnant."),
                  route=person("treatment_decision"))
    body = post(llm, "I am pregnnat and have pain, what can I take?", [kb])
    assert body["status"] == "escalated" and not llm.calls


def test_an_out_of_scope_reply_has_no_prefix():
    llm = FakeLLM(reply={"answer": "", "status": "out_of_scope", "citations": []}, route=person("none"))
    body = post(llm, "What is the active ingredient of Voltarin?")
    assert body["status"] == "out_of_scope" and body["answer"] == NOT_FOUND["en"]


def test_the_synthetic_fixture_has_the_words_used_above():
    assert "inflammation" in NSAIDS.render().lower() and "Voltaren" in NSAIDS.render()


# ------------------------------------------------------------------------------------- the eval cases

def _issue32_cases():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "evaluation" / "cases_issue32.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_issue32_cases_agree_with_the_self_care_rule():
    from medrag.rules import check
    from medrag.selfcare import is_self_care

    for case in _issue32_cases():
        cid, msg = case["id"], case["message"]
        if cid.startswith("typo32-ambiguous"):
            continue
        if cid.startswith("typo32-selfcare"):
            msg = msg.replace("headace", "headache")  # as the typo correction would
        if cid.startswith(("selfcare32-qualifier", "selfcare32-other-person", "selfcare32-dose", "selfcare32-emergency")):
            assert case["expected_status"] == "escalated" and not is_self_care(msg), cid
        elif "selfcare" in cid and case["expected_status"] == "answered":
            assert is_self_care(msg), cid
        if case["category"] in ("dose", "emergency"):
            assert check(msg) in ("dose", "emergency"), cid
