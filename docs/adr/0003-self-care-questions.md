# Answer first-person self-care questions with general KB information

Issue #32. A user typed "what can i take if i have headache" and got Out of Scope; "I have a headache, what can I take?" was Escalated as a Treatment Decision. The first was a bug (the support check treated "have" as a medical term; fixed in the same issue). The second was the safety policy working as written: a question from a specific person about what to take is a Treatment Decision, and a Treatment Decision is Escalated.

## Context

The policy until now was: choosing, starting, switching or stopping a drug for a specific person is the treating doctor's judgement, so any such question is Escalated. "I" is a specific person, so "I have a headache, what can I take?" was Escalated, even though the Knowledge Base states, for a headache, which Drug Classes and drugs are used (KB-03, Simple Analgesics and NSAIDs). For the everyday case (a headache, a fever, heartburn) with nothing risky in the message, the refusal gives the user nothing, and what they want is the same group-level fact the assistant already gives for "what is used for a headache?".

## Decision

A **Self-care Question** is a first-person question about the user's OWN symptom or condition that asks what they can take ("I have a headache, what can I take?", "عندي صداع اخد ايه", "انا عندي حموضة اخد ايه", "3andy sora3 a5od eh"). When it has **no risk qualifier** it is not a Treatment Decision:

- it is **Answered** with general Knowledge Base information: the drug classes and drugs the KB lists for that condition, with Citations, as information and not as advice for this person (no Dose, no "take this one");
- the server **appends a fixed note**, one sentence in the reply's language, after the model's answer, so it is always there and never left to the model: "A pharmacist can tell you which of these suits you." (Arabic and Franco-Arabic: "يمكن للصيدلي أن يخبرك أيها يناسبك.");
- a condition the KB does not cover is still Out of Scope ("I have malaria what can I take"): nothing is invented.

It stays **Escalated** when any qualifier is present (the list below), when it is about another specific person, when it asks for a Dose, or when it is an Emergency.

### The qualifier list

Matched on normalised text (`medrag/selfcare.py`), in English, Arabic, Egyptian dialect and Franco-Arabic. Each family is a set of regular expressions; the lists are deliberately broad (a word that is not a risk only turns a question that would have been Answered into an Escalated one, the safe direction).

| family | examples |
| --- | --- |
| pregnancy, breastfeeding | pregnant, expecting, nursing, trimester, حامل, رضاعة, 7amel |
| a child or an age under 18 | child, kid, baby, infant, toddler, teenager, "I am 14", "6 months old", طفل, رضيع, tefl |
| elderly or an age from 65 | elderly, senior, "I am 70", مسن, كبار السن |
| kidney or liver disease | kidney, renal, dialysis, liver, hepatic, cirrhosis, كلى, كبد |
| another chronic condition | diabetes, sugar, hypertension, blood pressure, asthma, COPD, heart, ulcer, epilepsy, thyroid, cancer, gout, cholesterol, stroke, pneumonia, tuberculosis, depression, "chronic" (only when the user says they have it; not migraine, arthritis, IBS or heartburn), سكر, ضغط, ربو, قلب, sokkar, daghT |
| other medicines being taken | "I am taking", "I take warfarin", "my medication", "instead of", "switch", "didn't work", blood thinners, insulin, باخد, بدل |
| an allergy | "allergic to penicillin", حساسية من |
| a severity or red-flag word | worst, sudden, severe, very bad, unbearable, blurred vision, blood, dizzy, جامد, شديد, مش طايق, فظيع, shedeed, awy |
| how long it has lasted | "for 3 days", "for weeks", "since last month", من ايام, بقالي اسبوع, men ayam |
| a Dose request | how much, how many, mg, dose, كام حباية, جرعة, kam 7aba, gor3a |
| another person | my wife, my son, my mother, my patient, he, she, ماما, ابني, عندها, mama, ebny, 3ando |
| an Emergency | everything `rules.check` flags (chest pain, cannot breathe, overdose, ...) |

### How it is implemented, and what it can never lower

- `medrag/selfcare.py` holds the rule: `is_self_care` is true only for a message that is first person, asks what to take (or what is good for something), and has no qualifier. A qualifier found anywhere keeps the escalation: in the message, in its typo-corrected form (`typos.py`), in the router's standalone rewrite, and in the earlier user turns of the Session ("my son has a fever" ... "what can I take?").
- `pipeline.decide` uses the rule for exactly one thing that lowers: the router's **`treatment_decision`** becomes none. It never lowers an emergency, a dose or a diagnosis. The same module also RAISES: a first-person ask for what to take that carries a qualifier ("عندي صداع جامد اخد ايه", "I take warfarin, what can I take for a headache?") is a Treatment Decision even when the router said none, because the policy says such a question stays Escalated and a router miss must not turn it into an answer. The fixed rules (`rules.check`) still run first on the raw message and on the corrected one, and nothing can lower what they find. The invariant "escalation can only rise" holds for everything except this one named case.
- The router prompt (`medrag/routing.py`) states the exception in one clause of the definition of `treatment_decision` (a longer text with examples changed the 4B router's rewrites of follow-up questions, so it is kept to one clause: the fixed rules, not the router, enforce the policy); the answer call's backstop prompt says the same, and a backstop `treatment_decision` is ignored for a self-care question (a backstop emergency, dose or diagnosis is not).
- The answer call gets a short note telling it to give general information with no dose and no recommendation of one drug; the dose check on the reply is unchanged.

## Considered options

- **Keep escalating everything first-person (the old policy).** Safest, but it refuses the most common real question, and the author asked for the change.
- **Let the router alone decide.** The 4B router is inconsistent on exactly these borderline messages; the qualifier list has to be a fixed, tested rule, and the router may only escalate more.
- **Add the note in the prompt.** A model can drop it; the server appends it.
- **Answer even with a qualifier, with a stronger note.** Rejected: pregnancy, children, kidney disease, other medicines and red flags are the cases where a general list is most likely to mislead.

## Consequences

- A user who words a risk in a way the lists do not know ("I'm 8 months along", a drug class not listed, an unusual dialect word) gets a general answer plus the pharmacist note instead of an escalation. The answer is only what the KB states for the condition and never a Dose; this is the accepted residual risk of the policy. New wordings are added to `selfcare.py` with a test.
- The lists over-escalate: a word such as "chest", "blood", "condition" or "for days" anywhere in the message escalates it. That is intended.
- Escalation of other-person questions is unchanged ("my wife has a headache, what can she take?" is still a Treatment Decision).
- The evaluation has a new set (`evaluation/cases_issue32.json`) with the self-care wordings, each qualifier family, other-person and Dose variants, and the Out of Scope cases; the older sets contained no first-person self-care question, which is why this was not caught.

## Revision 1 (review round after the first implementation)

Found by review on the live stack and decided by the author:

- **A lookup is not self-care.** Asking what is used FOR a condition is a normal drug-information question, even in the first person
  ("Which drugs can I use for asthma?", "What can I take for migraine?", "What can I use for arthritis pain?", "Which NSAID can I use
  in patients with heart failure?", "What can a pregnant woman take for her heartburn?", "If a patient has asthma, what can he take for
  pain?"). The users are pharmacists and doctors. These are Answered (Out of Scope when the KB lacks the condition) and never
  escalated; the router's `treatment_decision` is lowered for them. They get no pharmacist note.
- **A condition is a qualifier only when the user says they HAVE it** ("I have / I'm / my X", "عندي X", "3andy X") and it is not the
  condition being asked about. "I have asthma and a headache, what can I take?" escalates; "What can I take for asthma?" does not.
  Pregnancy, breastfeeding, a child, an elderly person and other medicines likewise count when stated about oneself; group phrasings
  ("in pregnancy", "a pregnant woman") stay lookups (GLOSSARY). Migraine, arthritis, IBS and GERD / heartburn are everyday
  indications the KB covers and are off the chronic list; the words "chronic" and "long-term" still qualify.
- **Another specific person** is a relative ("my wife", "my son", "my patient") or a pronoun with no generic noun before it; a pronoun
  after a generic noun ("a patient ... he", "a pregnant woman ... her") is a group.
- **Invariant.** A first-person strict "what can I take / use" ask is a lookup (answered), a self-care question (answered with the
  note) or escalated; when the user states their own symptom it is never a plain answer. A qualifier found in the message, its
  corrected form, the router's rewrite or an earlier turn of the Session escalates it even when the router says none (the first
  version only blocked self-care there and fell through to a plain answer: "I'm pregnant" then "I have a headache, what can I take?"
  was answered with NSAIDs).
- **Red flags and misspellings.** Severity patterns now include weak\w*, a stiff neck (either word order, Arabic ناشفة / متخشبة), a
  temperature of 39 C / 103 F or more and alcohol; qualifier words within one edit of a short keyword list ("pregnent", "sever",
  "sevear", "childern") are read as the keyword.
- **Dose-shaped asks** ("what can I take and how many / how often") get the Dose trigger and the Dose text.
- **No prescription marking.** The self-care note stays the one fixed sentence (the author reversed an earlier idea to name the
  prescription-only items).
