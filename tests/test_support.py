"""Out of Scope and Citation checking (#5): snippet repair to the closest real line, a Citation must
be a retrieved Section that supports the question, partial answers stay Answered. Synthetic Sections
and a fake LLM only (no KB, no Ollama); the real-KB check at the end skips without the .docx files."""
import asyncio
import json
from pathlib import Path

import pytest
from synthetic import FakeLLM

from medrag.answering import NOT_FOUND, SYSTEM_PROMPT, answer_question, build_citations
from medrag.gate import LLMGate
from medrag.models import Line, Row, Section, Table
from medrag.retrieval import LexicalRetriever
from medrag.support import SupportCheck

PLAQUENIL_ROW = "Plaquenil – Hydroxychloroquine"


def products(*rows):
    return Table(Row.from_cells(["Trade name", "Active ingredient"]), tuple(Row.from_cells(list(r)) for r in rows),
                 "products", 0, 1)


DMARD = Section(
    "KB-09", "Arthritis", "Conventional DMARDs", "class",
    (Line("الاستخدام: علاج الروماتويد وتقليل الالتهاب."),
     products(("Plaquenil", "Hydroxychloroquine"), ("Arava", "Leflunomide")),
     Line("Common side effects:"), Line("Nausea and diarrhea."), Line("Rash.")),
)
TETRA = Section(
    "KB-05", "Pneumonia", "Tetracyclines", "class",
    (Line("الاستخدام: علاج بعض حالات الالتهاب الرئوي البكتيري، ويُستخدم Doxycycline في حالات مختارة."),
     products(("Vibramycin", "Doxycycline")),
     Line("Contraindications / أهم التحذيرات:"),
     Line("Pregnancy: يُتجنب عادةً، إلا لو الطبيب قرر غير ذلك."),
     Line("الحذر في الأطفال أقل من 8 سنوات.")),
)
SABA = Section(
    "KB-01", "Asthma", "SABA", "class",
    (Line("Common side effects:"), Line("Tremors."), Line("Palpitations and tachycardia."),
     products(("Ventolin", "Salbutamol"))),
)
PPI = Section("KB-02", "Peptic Ulcer", "PPIs", "class", (products(("Controloc", "Pantoprazole")),))
KB = [DMARD, TETRA, SABA, PPI]
SUPPORT = SupportCheck(KB)
PREGNANCY_LINE = "Pregnancy: يُتجنب عادةً، إلا لو الطبيب قرر غير ذلك."


def reply(answer, section, snippet):
    return {"answer": answer, "status": "answered",
            "citations": [{"section_id": section.section_id, "snippet": snippet}]}


def ask(question, raw, sections=KB, language="en"):
    gate = LLMGate(FakeLLM(raw))
    return asyncio.run(answer_question(question, language, sections, gate, support=SUPPORT)).model_dump()


# ------------------------------------------------------------------ snippet repair

REPAIR = Section(
    "KB-03", "Headache", "NSAIDs", "class",
    (Line("Common side effects:"),
     Line("Nausea and stomach upset."),
     Line("Dizziness, headache and rash on the skin."),
     Line("Contraindications"),
     Line("Pregnancy: avoided in the third trimester.")),
)


def repaired(given, hints="q"):
    (c,) = build_citations([{"section_id": REPAIR.section_id, "snippet": given}], [REPAIR], hints)
    return c.snippet


def test_a_paraphrased_snippet_is_repaired_to_the_closest_real_line():
    assert repaired("Patients may feel dizzy and develop a skin rash") == "Dizziness, headache and rash on the skin."
    assert repaired("upset stomach and nausea") == "Nausea and stomach upset."


def test_repair_follows_the_paraphrase_not_the_question():
    # The question's words favour the nausea line; the model's own words are about the pregnancy line.
    hints = "What are the side effects, nausea and stomach upset? Nausea and stomach upset occur."
    assert repaired("do not use in the third trimester of pregnancy", hints) == "Pregnancy: avoided in the third trimester."


def test_repair_never_picks_a_heading_or_label_line():
    # Words of the paraphrase match only the label and the bare heading word: a content line is chosen.
    for given in ("common side effects of the drug", "contraindications"):
        assert repaired(given, hints="drug") in REPAIR.lines[1:2] + REPAIR.lines[2:3] + REPAIR.lines[4:]
    assert repaired("NSAIDs Headache") in REPAIR.lines  # the heading line is never offered


def test_a_made_up_citation_section_is_dropped_and_the_reply_downgraded():
    raw = {"answer": "Diclofenac.", "status": "answered",
           "citations": [{"section_id": "KB-99 › nowhere", "snippet": "Cataflam – Diclofenac"}]}
    body = ask("What is the active ingredient of Controloc?", raw)
    assert body["status"] == "out_of_scope" and body["citations"] == []
    assert body["answer"] == NOT_FOUND["en"]


def test_a_section_that_was_not_retrieved_cannot_be_cited():
    raw = reply("Pantoprazole.", PPI, "Controloc – Pantoprazole")
    body = ask("What is the active ingredient of Controloc?", raw, sections=[DMARD, TETRA])
    assert body["status"] == "out_of_scope" and body["citations"] == []


# ------------------------------------------------------------------ support check

def test_malaria_trap_real_rows_cited_for_a_claim_the_rows_do_not_make():
    raw = reply("Plaquenil (Hydroxychloroquine) is used to treat malaria.", DMARD, PLAQUENIL_ROW)
    for q in ("Which drugs are used to treat malaria?", "Is Plaquenil used for malaria?",
              "Can Plaquenil be used to prevent malaria when travelling?"):
        body = ask(q, raw)
        assert body["status"] == "out_of_scope" and body["citations"] == [], q
        assert body["answer"] == NOT_FOUND["en"]


def test_malaria_trap_in_arabic_gets_the_arabic_fixed_text():
    raw = reply("Plaquenil بيعالج الملاريا.", DMARD, PLAQUENIL_ROW)
    body = ask("هل Plaquenil بيعالج الملاريا؟", raw, language="ar")
    assert body["status"] == "out_of_scope" and body["citations"] == [] and body["answer"] == NOT_FOUND["ar"]


def test_a_dose_in_an_unsupported_answer_is_still_escalated_not_out_of_scope():
    from medrag.escalation import ESCALATION_TEXT
    raw = reply("Take 500 mg twice a day for malaria.", DMARD, PLAQUENIL_ROW)
    body = ask("Is Plaquenil used for malaria?", raw)
    assert body["status"] == "escalated" and body["answer"] == ESCALATION_TEXT["dose"]["en"]
    assert body["citations"] == []


def test_a_model_that_cites_a_drug_section_for_an_off_topic_request_is_downgraded():
    raw = reply("Dear hiring manager, ...", PPI, "Controloc – Pantoprazole")
    body = ask("Write me a cover letter for a job application.", raw)
    assert body["status"] == "out_of_scope" and body["citations"] == []


@pytest.mark.parametrize("question, raw", [
    ("What is the active ingredient of Controloc?", reply("Pantoprazole.", PPI, "Controloc – Pantoprazole")),
    ("Which drug is in Ventolin?", reply("Salbutamol.", SABA, "Ventolin – Salbutamol")),
    ("What are the side effects of salbutamol?", reply("Tremors, palpitations.", SABA, "Tremors.")),
    ("Is Doxycycline suitable for children under 8 years?",
     reply("Use with caution under 8.", TETRA, "الحذر في الأطفال أقل من 8 سنوات.")),
    ("Doxycycline in pregnancy?", reply("It is usually avoided.", TETRA, PREGNANCY_LINE)),
    ("هل Doxycycline ينفع في الحمل؟", reply("يُتجنب عادةً.", TETRA, PREGNANCY_LINE)),
])
def test_legitimate_answers_are_not_downgraded(question, raw):
    body = ask(question, raw)
    assert body["status"] == "answered" and len(body["citations"]) == 1, body


@pytest.mark.parametrize("question, raw", [
    ("What is the active ingredient of Controloc and how much does it cost in Egypt?",
     reply("Controloc contains Pantoprazole. The price is not in the knowledge base.", PPI, "Controloc – Pantoprazole")),
    ("What are the side effects of salbutamol, and is it safe with alcohol?",
     reply("Tremors and palpitations. Alcohol is not covered by the knowledge base.", SABA, "Tremors.")),
    ("إيه المادة الفعالة في Controloc وسعره كام؟",
     reply("Pantoprazole. السعر غير موجود في قاعدة المعرفة.", PPI, "Controloc – Pantoprazole")),
])
def test_partly_covered_questions_stay_answered(question, raw):
    body = ask(question, raw)
    assert body["status"] == "answered" and body["citations"][0]["snippet"] in {"Controloc – Pantoprazole", "Tremors."}


def test_partial_answer_with_a_repaired_snippet_is_still_answered():
    raw = reply("Pantoprazole. The price is not in the knowledge base.", PPI, "Controloc contains pantoprazole")
    body = ask("What is the active ingredient of Controloc and what is its price?", raw)
    assert body["status"] == "answered" and body["citations"][0]["snippet"] == "Controloc – Pantoprazole"


def test_topic_terms_ignore_frame_words_drug_names_and_numbers():
    assert SUPPORT.topic_terms("Which drugs are used to treat malaria?") == ["malaria"]
    assert SUPPORT.topic_terms("Is Plaquenil used for malaria in 2 patients?") == ["malaria"]
    assert SUPPORT.topic_terms("What is the active ingredient of Controloc?") == []


def test_matching_is_morphology_and_bilingual_aware():
    sec = Section("KB-08", "Hypertension", "ARBs", "class",
                  (Line("Antihypertensive drugs. Tremor and Pregnancy: avoided."),))
    check = SupportCheck([sec])
    assert check.supported("Which drugs lower hypertension?", [sec])
    assert check.supported("Does it cause tremors?", [sec])
    assert check.supported("ممنوع في الحمل؟", [sec])
    assert not check.supported("Which drugs are used to treat malaria?", [sec])


def test_prompt_forbids_outside_knowledge_and_asks_for_partial_answers():
    assert "Never use outside knowledge" in SYSTEM_PROMPT
    assert "ANYTHING that answers part of the question" in SYSTEM_PROMPT
    assert "ONLY when nothing that was asked" in SYSTEM_PROMPT


# ------------------------------------------------------------------ real KB (skipped without the .docx files)

CASES = json.loads((Path(__file__).resolve().parents[1] / "evaluation" / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.kb
def test_real_kb_no_expected_answer_is_downgraded_and_every_trap_is_caught(real_kb):
    sections = [s for d in real_kb.values() for s in d.sections]
    check, search = SupportCheck(sections), LexicalRetriever(sections).search
    wrongly_downgraded, missed = [], []
    for case in CASES:
        for turn in case.get("turns") or [case]:
            msg, expected, top = turn["message"], turn.get("expected_status"), search(turn["message"], 4)
            if expected == "answered" and top:
                kb_ids = case.get("expected_kb_ids") or []
                cited = [s for s in top if not kb_ids or s.kb_id in kb_ids] or top
                if not check.supported(msg, cited):
                    wrongly_downgraded.append(case["id"])
            elif expected == "out_of_scope" and case["category"] in ("trap_out_of_scope", "off_topic", "dev") and top:
                if check.supported(msg, top):
                    missed.append(case["id"])
    assert wrongly_downgraded == []
    assert missed == []
