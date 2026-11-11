"""Self-care questions and lookups (ADR 0003): the deterministic policy in medrag/selfcare.py."""
import pytest

from medrag.rules import check
from medrag.selfcare import asks_what_to_take, is_self_care, judge, pharmacist_note, qualifiers, specific_other

SELF_CARE = [
    "what can i take if i have headache",
    "what i can take if i have headace",
    "I have a headache, what can I take?",
    "I have a sore throat. What is good for it?",
    "I have heartburn, which medicine can I take?",
    "I have migraine, what can I take?",
    "I have arthritis pain what can I take",
    "I have a headache, what can I take to stop it?",
    "My headache started this morning, what can I take?",
    "I get headaches, what can I take for them?",
    "I'm on my period and have cramps, what can I take?",
    "عندي صداع اخد ايه",
    "انا عندي حموضة اخد ايه",
    "3andy sora3 a5od eh",
    "3andy so5oneya a5od eh",
]

# (message, the qualifier family that must be found)
QUALIFIED = [
    ("I'm pregnant and have a headache, what can I take?", "pregnancy"),
    ("I have a headache and I am breastfeeding, what can I take?", "pregnancy"),
    ("I have a headache, I'm pregnent, what can I take?", "pregnancy"),
    ("What can I take while pregnant?", "pregnancy"),
    ("عندي صداع وانا حامل اخد ايه", "pregnancy"),
    ("عندي صداع وبرضع اخد ايه", "pregnancy"),
    ("I'm 15 and have a headache what can I take", "age"),
    ("I have a headache, I am 12 years old, what can I take?", "age"),
    ("I am 70 and have a headache what can I take", "age"),
    ("I'm elderly and have a headache, what can I take?", "elderly"),
    ("I'm a teenager with a headache, what can I take?", "child"),
    ("I have kidney disease and a headache, what can I take?", "kidney_liver"),
    ("I have a liver problem and a headache, what can I take?", "kidney_liver"),
    ("عندي مشاكل في الكبد وصداع اخد ايه", "kidney_liver"),
    ("I'm diabetic, I have a headache what can I take", "chronic"),
    ("I have asthma and a headache, what can I take?", "chronic"),
    ("I have high blood pressure and a headache, what can I take", "chronic"),
    ("I have a headache from my chronic condition, what can I take", "chronic"),
    ("عندي سكر وصداع اخد ايه", "chronic"),
    ("عندي صداع والضغط عالي اخد ايه", "chronic"),
    ("3andy sokkar w sora3 a5od eh", "chronic"),
    ("I take warfarin, what can I take for a headache?", "other_medicines"),
    ("I'm on warfarin and have a headache what can I take", "other_medicines"),
    ("I am taking Concor and have a headache, what can I take", "other_medicines"),
    ("I have a headache what can I take instead of Brufen", "other_medicines"),
    ("عندي صداع وباخد ادوية الضغط اخد ايه", "other_medicines"),
    ("I am allergic to penicillin and have a sore throat what can I take", "allergy"),
    ("I have a severe headache what can I take", "severity"),
    ("I have a sever headache what can I take", "severity"),
    ("I have a sevear headache what can I take", "severity"),
    ("I have the worst headache of my life, what can I take", "severity"),
    ("I have a headache for 3 days, what can I take?", "severity"),
    ("I have had a headache for weeks, what can I take?", "severity"),
    ("I have a headache and blurred vision, what can I take?", "severity"),
    ("I have a headache and my arm feels weak what can I take", "severity"),
    ("I have a headache and my neck is stiff, what can I take", "severity"),
    ("I have a headache and my stiff neck hurts, what can I take", "severity"),
    ("عندي صداع ورقبتي ناشفة اخد ايه", "severity"),
    ("عندي صداع ورقبتي متخشبة اخد ايه", "severity"),
    ("I have a headache and fever of 104F what can I take", "severity"),
    ("I have a headache and a temperature of 39.5 C what can I take", "severity"),
    ("I have a headache I drank alcohol last night what can I take", "severity"),
    ("عندي صداع وانا سكران اخد ايه", "severity"),
    ("عندي صداع جامد اخد ايه", "severity"),
    ("عندي صداع من ايام اخد ايه", "severity"),
    ("3andy sora3 shedeed a5od eh", "severity"),
    ("3andy sora3 men ayam a5od eh", "severity"),
    ("how many mg can I take for a headache", "dose"),
    ("I have a headache what can I take and how many", "dose"),
    ("عندي صداع كام حباية اخد", "dose"),
    ("3andy sora3 a5od kam 7aba", "dose"),
    ("my wife has a headache what can she take", "other_person"),
    ("I have a headache what can I give my son", "other_person"),
    ("my patient has a headache what should I give", "other_person"),
    ("ماما عندها صداع تاخد ايه", "other_person"),
    ("ebny 3ando sora3 edeeh eh", "other_person"),
    ("he has a fever, what can he take", "other_person"),
]

# What the author decided is a lookup: asking what is used FOR a condition is answered, never escalated.
LOOKUPS = [
    "Which drugs can I use for asthma?",
    "Which drugs can I use for COPD?",
    "Which drugs can I use for GERD?",
    "Which drugs can I use for diabetes?",
    "Which drugs can I use for hypertension?",
    "What antibiotic can I use for pneumonia?",
    "What can I take for migraine?",
    "What can I use for arthritis pain?",
    "What can I take for IBS?",
    "Which drugs can I use for pharyngitis?",
    "What can I use for hypertension?",
    "Which NSAID can I use in patients with heart failure?",
    "What can I use to treat heartburn in pregnancy?",
    "What can a pregnant woman take for her heartburn?",
    "If a patient has asthma, what can he take for pain?",
    "اخد ايه للضغط",
    "اخد ايه للربو",
]


@pytest.mark.parametrize("message", SELF_CARE)
def test_a_first_person_own_symptom_ask_without_a_qualifier_is_self_care(message):
    assert asks_what_to_take(message)
    assert qualifiers(message) == []
    v = judge(message)
    assert v.self_care and v.lower and v.raise_to is None
    assert is_self_care(message)


@pytest.mark.parametrize("message,family", QUALIFIED)
def test_every_qualifier_family_is_found_and_escalates(message, family):
    assert family in qualifiers(message)
    v = judge(message)
    assert not v.self_care and not v.lower
    if family == "dose":
        assert v.raise_to == "dose" or check(message) == "dose"  # the fixed rules catch the plain wordings
    elif asks_what_to_take(message) or family == "other_person":
        assert v.raise_to == "treatment_decision"
    assert not is_self_care(message)


@pytest.mark.parametrize("message", LOOKUPS)
def test_asking_what_is_used_for_a_condition_is_a_lookup_never_escalated(message):
    assert qualifiers(message) == []
    v = judge(message)
    assert v.raise_to is None and v.lower and not v.self_care  # answered, no pharmacist note


def test_a_condition_the_user_has_is_a_qualifier_but_the_condition_asked_about_is_not():
    assert "chronic" not in qualifiers("What can I take for asthma?")
    assert "chronic" not in qualifiers("What can I take for my asthma?")
    assert "chronic" in qualifiers("I have asthma and a headache, what can I take?")
    assert "chronic" in qualifiers("I have a headache, what can I take? I'm diabetic")
    assert "chronic" not in qualifiers("I have a headache, what can I take for my asthma?")
    assert "pregnancy" not in qualifiers("Is Brufen safe in pregnancy?")
    assert "pregnancy" not in qualifiers("What can a pregnant woman take for her heartburn?")
    assert "pregnancy" in qualifiers("I'm pregnant, what can I take for heartburn?")


@pytest.mark.parametrize("message", ["I have migraine, what can I take?", "I have IBS, what can I take?", "I have GERD, what can I take?"])
def test_migraine_arthritis_ibs_and_gerd_are_everyday_indications_not_chronic_qualifiers(message):
    assert qualifiers(message) == []


def test_chronic_and_long_term_always_qualify():
    assert "chronic" in qualifiers("I have a chronic headache, what can I take?")
    assert "chronic" in qualifiers("I have a long-term headache, what can I take?")


@pytest.mark.parametrize("message", [
    "What is Brufen?", "Is Brufen safe in pregnancy?", "What are the side effects of Brufen in children?",
    "What medicines interact with my warfarin?", "Which drug class is my medicine in?", "Write me a poem",
])
def test_a_message_that_is_not_a_first_person_ask_is_left_to_the_router(message):
    v = judge(message)
    assert v.raise_to is None and not v.self_care


@pytest.mark.parametrize("text,expected", [
    ("my wife has a headache", True), ("my patient has gout", True), ("she has a fever", True), ("he took too much", True),
    ("Can I give my 5-year-old Doxycycline for his sinusitis?", True),
    ("If a patient has asthma, what can he take for pain?", False),
    ("What can a pregnant woman take for her heartburn?", False),
    ("What can patients with asthma take?", False),
    ("ماما عندها صداع", True), ("مريض عنده سكر ياخد ايه", False), ("ebny 3ando sora3", True), ("mareed 3ando sokkar", False),
])
def test_specific_other_person_vs_a_group_with_a_generic_noun(text, expected):
    assert specific_other(text) is expected


@pytest.mark.parametrize("message", [
    "I have severe chest pain and cold sweat, what can I take?", "I can't breathe, what can I take?",
    "I swallowed a whole box of Panadol, what can I take?", "I want to die, what can I take?",
])
def test_an_emergency_is_never_self_care(message):
    assert check(message) == "emergency"
    assert not is_self_care(message)


@pytest.mark.parametrize("message", [
    "How many mg of Brufen can I take, I have a headache?", "I have a headache, how often can I take Panadol?",
])
def test_a_dose_request_is_never_self_care(message):
    assert check(message) == "dose"
    assert not is_self_care(message)


def test_a_qualifier_in_the_corrected_form_the_rewrite_or_an_earlier_turn_escalates():
    msg = "I have a headache, what can I take?"
    assert judge(msg).self_care
    assert judge(msg, "I have a headache, I am pregnant, what can I take?").raise_to == "treatment_decision"
    assert judge(msg, rewrite="What can I take for a severe headache?").raise_to == "treatment_decision"
    assert judge(msg, rewrite="What can a pregnant woman take for a headache?").raise_to is None  # a group phrasing
    assert judge(msg, context=("I'm pregnant",)).raise_to == "treatment_decision"
    assert judge(msg, context=("I am breastfeeding my baby",)).raise_to == "treatment_decision"
    assert judge(msg, context=("I take warfarin every day",)).raise_to == "treatment_decision"
    assert judge(msg, context=("I have kidney disease",)).raise_to == "treatment_decision"
    assert judge("what can I take for it?", context=("my son has a fever",)).raise_to == "treatment_decision"
    assert judge(msg, context=("What is Brufen?",)).self_care


def test_the_invariant_a_strict_ask_with_the_users_own_symptom_is_self_care_or_escalated_never_plain():
    for message in [m for m, _ in QUALIFIED] + SELF_CARE:
        if not message or not any(w in message.lower() for w in ("i have", "i'm", "عندي", "3andy", "i take", "i am", "i get", "my ")):
            continue
        from medrag.selfcare import asks_strictly

        if not asks_strictly(message):
            continue
        v = judge(message)
        assert v.self_care or v.raise_to, message


def test_the_ask_may_be_in_the_corrected_form():
    assert judge("waht cna i tke if i have headace", "what can i take if i have headache").self_care


def test_a_dose_shaped_ask_gets_the_dose_trigger():
    assert judge("I have a headache what can I take and how often").raise_to == "dose"
    assert judge("I have a headache, what can I take? Also tell me max per day").raise_to == "dose"


def test_a_misspelled_qualifier_is_still_a_qualifier():
    for typo, family in [("pregnent", "pregnancy"), ("pregant", "pregnancy"), ("sever", "severity"), ("sevear", "severity"),
                         ("diabtes", "chronic")]:
        msg = f"I have a headache and I'm {typo}, what can I take?" if family == "pregnancy" else (
            f"I have a {typo} headache what can I take" if family == "severity" else f"I have {typo} and a headache, what can I take?")
        assert family in qualifiers(msg), msg
        assert judge(msg).raise_to == "treatment_decision", msg


def test_the_pharmacist_note_is_one_sentence_in_the_reply_language():
    assert pharmacist_note("en") == "A pharmacist can tell you which of these suits you."
    assert pharmacist_note("ar") == pharmacist_note("mixed") and "صيدل" in pharmacist_note("ar")
