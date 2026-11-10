"""Issue 32 against the real KB (skipped when the .docx files are absent): the repro questions pass the support check on
KB-03, a typo is corrected to a real KB word, and no existing eval question is rewritten by the typo correction."""
import json
from pathlib import Path

import pytest

from medrag.language import detect_language
from medrag.support import SupportCheck
from medrag.typos import TypoCorrector

pytestmark = pytest.mark.kb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kb(real_kb):
    sections = [s for d in real_kb.values() for s in d.sections]
    kb03 = [s for s in real_kb["KB-03"].sections if "Simple Analgesics" in s.render() or "NSAIDs" in s.render()]
    return sections, SupportCheck(sections), TypoCorrector(sections), kb03


@pytest.mark.parametrize("question", [
    "what can i take if i have headache",
    "I have a headache, what can I take?",
    "what i can take if i have headache",
])
def test_the_repro_questions_are_supported_by_the_kb03_sections(kb, question):
    _, support, _, kb03 = kb
    assert kb03
    assert support.supported(question, kb03)


def test_a_misspelled_condition_is_corrected_to_a_kb_word_and_then_supported(kb):
    _, support, typos, kb03 = kb
    fixed = typos.correct("what i can take if i have headace")
    assert fixed.text == "what i can take if i have headache" and fixed.words == ["headache"]
    assert support.supported(fixed.text, kb03)
    assert not support.supported("what i can take if i have headace", kb03)  # why the correction is needed


def test_a_misspelled_brand_is_corrected_and_an_unknown_condition_is_not(kb):
    _, _, typos, _ = kb
    assert typos.correct("Brufin alternatives").text == "Brufen alternatives"
    assert typos.correct("I have malaria what can I take").changes == ()


def test_no_eval_question_is_rewritten_by_the_typo_correction(kb):
    _, _, typos, _ = kb
    for f in sorted((ROOT / "evaluation").glob("cases*.json")):
        for case in json.loads(f.read_text(encoding="utf-8")):
            for turn in case.get("turns", [case]):
                msg = turn["message"]
                if case["id"].startswith("typo32"):  # the new cases that are meant to be corrected
                    continue
                assert typos.correct(msg, detect_language(msg)).changes == (), (case["id"], msg)
