"""Self-care questions (ADR 0003): the deterministic rule, its qualifier families and the invariant that it can never
lower an Emergency, a Dose request or a diagnosis."""
import pytest

from medrag.rules import check
from medrag.selfcare import asks_what_to_take, is_self_care, pharmacist_note, qualifiers

SELF_CARE = [
    "what can i take if i have headache",
    "what i can take if i have headace",
    "I have a headache, what can I take?",
    "I have a sore throat. What is good for it?",
    "I have heartburn, which medicine can I take?",
    "عندي صداع اخد ايه",
    "انا عندي حموضة اخد ايه",
    "عندي برد ايه الدوا",
    "3andy sora3 a5od eh",
    "3andy so5oneya a5od eh",
]

# (message, the qualifier family that must be found)
QUALIFIED = [
    ("I'm pregnant and have a headache, what can I take?", "pregnancy"),
    ("What can I take for a headache? I'm breastfeeding", "pregnancy"),
    ("عندي صداع وانا حامل اخد ايه", "pregnancy"),
    ("my child has a fever what can I take", "child"),
    ("I'm 15 and have a headache what can I take", "age"),
    ("I have a headache, I am 12 years old, what can I take?", "age"),
    ("my baby is 6 months old", "age"),
    ("I am 70 and have a headache what can I take", "age"),
    ("I'm elderly and have a headache, what can I take?", "elderly"),
    ("I have kidney disease and a headache, what can I take?", "kidney_liver"),
    ("عندي مشاكل في الكبد وصداع اخد ايه", "kidney_liver"),
    ("I'm diabetic, I have a headache what can I take", "chronic"),
    ("I have high blood pressure and a headache, what can I take", "chronic"),
    ("I have asthma and a cough what can I take", "chronic"),
    ("عندي سكر وصداع اخد ايه", "chronic"),
    ("3andy sokkar w sora3 a5od eh", "chronic"),
    ("I take warfarin, what can I take for a headache?", "other_medicines"),
    ("I am taking Concor and have a headache, what can I take", "other_medicines"),
    ("I have a headache what can I take instead of Brufen", "other_medicines"),
    ("عندي صداع وباخد ادوية الضغط اخد ايه", "other_medicines"),
    ("I am allergic to penicillin and have a sore throat what can I take", "allergy"),
    ("I have a severe headache what can I take", "severity"),
    ("I have the worst headache of my life, what can I take", "severity"),
    ("I have a sudden headache, what can I take?", "severity"),
    ("I have a headache for 3 days, what can I take?", "severity"),
    ("I have had a headache for weeks, what can I take?", "severity"),
    ("I have a headache and blurred vision, what can I take?", "severity"),
    ("عندي صداع جامد اخد ايه", "severity"),
    ("عندي صداع مش طايقه اخد ايه", "severity"),
    ("عندي صداع من ايام اخد ايه", "severity"),
    ("3andy sora3 shedeed a5od eh", "severity"),
    ("3andy sora3 men ayam a5od eh", "severity"),
    ("how many mg can I take for a headache", "dose"),
    ("I have a headache, what dose can I take?", "dose"),
    ("عندي صداع كام حباية اخد", "dose"),
    ("3andy sora3 a5od kam 7aba", "dose"),
    ("my wife has a headache what can she take", "other_person"),
    ("I have a headache what can I give my son", "other_person"),
    ("my patient has a headache what should I give", "other_person"),
    ("ماما عندها صداع تاخد ايه", "other_person"),
    ("ebny 3ando sora3 edeeh eh", "other_person"),
    ("he has a fever, what can he take", "other_person"),
]


@pytest.mark.parametrize("message", SELF_CARE)
def test_a_first_person_question_without_a_qualifier_is_self_care(message):
    assert asks_what_to_take(message)
    assert qualifiers(message) == []
    assert is_self_care(message)


@pytest.mark.parametrize("message,family", QUALIFIED)
def test_every_qualifier_family_keeps_a_question_out_of_self_care(message, family):
    assert family in qualifiers(message)
    assert not is_self_care(message)


@pytest.mark.parametrize("message", [
    "Is Brufen safe in pregnancy?", "What is the active ingredient in Panadol?", "Which drugs treat hypertension?",
    "Can I take Panadol at 3pm?", "What can you tell me about NSAIDs?", "Write me a poem",
])
def test_a_message_that_does_not_ask_what_to_take_is_never_self_care(message):
    assert not is_self_care(message)


def test_the_message_must_be_first_person():
    assert not is_self_care("what can he take for a headache")
    assert not is_self_care("what should the patient take for a headache")


@pytest.mark.parametrize("message", [
    "I have severe chest pain and cold sweat, what can I take?",
    "I can't breathe, what can I take?",
    "I swallowed a whole box of Panadol, what can I take?",
    "My daughter is having a seizure what can I take",
    "عندي ضيق نفس شديد اخد ايه",
    "I want to die, what can I take?",
])
def test_an_emergency_is_never_self_care(message):
    assert check(message) == "emergency"
    assert not is_self_care(message)


@pytest.mark.parametrize("message", [
    "How many mg of Brufen can I take, I have a headache?",
    "I have a headache, how often can I take Panadol?",
    "I have a headache, what is the dose of Panadol I can take?",
])
def test_a_dose_request_is_never_self_care(message):
    assert check(message) == "dose"
    assert not is_self_care(message)


def test_a_qualifier_in_the_corrected_or_rewritten_form_or_in_an_earlier_turn_keeps_the_escalation():
    msg = "what can i take if i have headache"
    assert is_self_care(msg)
    assert not is_self_care(msg, "what can i take if i have headache while pregnant")
    assert not is_self_care(msg, context=("my son has a fever",))
    assert not is_self_care(msg, context=("I am pregnant",))
    assert is_self_care(msg, "what can i take if i have a headache", context=("What is Brufen?",))


def test_the_ask_may_be_in_the_corrected_form():
    assert is_self_care("waht cna i tke if i have headace", "what can i take if i have headache")


def test_the_pharmacist_note_is_one_sentence_in_the_reply_language():
    assert pharmacist_note("en") == "A pharmacist can tell you which of these suits you."
    assert pharmacist_note("ar") == pharmacist_note("mixed") and "صيدل" in pharmacist_note("ar")


# ------------------------------------------------------------ a qualifier RAISES a self-care attempt (router said none)

@pytest.mark.parametrize("message,family", [q for q in QUALIFIED if asks_what_to_take(q[0])])
def test_a_first_person_ask_with_a_qualifier_is_escalated_even_when_the_router_says_none(message, family):
    from medrag.selfcare import escalates_with_qualifier

    assert escalates_with_qualifier(message)


def test_escalates_with_qualifier_needs_an_ask_for_what_to_take_and_a_qualifier():
    from medrag.selfcare import escalates_with_qualifier

    assert escalates_with_qualifier("عندي صداع جامد اخد ايه")
    assert escalates_with_qualifier("I have a headache and I'm pregnant, what can I take?")
    assert escalates_with_qualifier("3andy sora3 men ayam a5od eh")
    assert escalates_with_qualifier("what i can take if i have headace pregnnat", "what i can take if i have headache pregnant")
    assert not escalates_with_qualifier("I have a headache, what can I take?")
    assert not escalates_with_qualifier("Is Brufen safe in pregnancy?")  # no ask for what to take: left to the router
    assert not escalates_with_qualifier("What are the side effects of Brufen in children?")
